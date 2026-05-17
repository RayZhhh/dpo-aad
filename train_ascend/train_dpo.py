import os
import argparse
import pickle
import sys
import types
import json
import gc


from datasets import load_dataset, Dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel

from utils import (
    is_ascend_available,
    set_device,
    is_bf16_supported,
    get_device_for_rank,
    get_dtype,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_merged_model", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to save memory",
    )
    return p.parse_args()


def longest_seq_len(dataset, tok):
    # Apply chat template to convert messages to string before tokenizing
    def apply_chat_template(example, key):
        messages = example[key]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return max(
        max(
            len(tok(apply_chat_template(example, "chosen")).input_ids),
            len(tok(apply_chat_template(example, "rejected")).input_ids),
        )
        for example in dataset
    )


def longest_prompt_len(dataset, tok):
    return max(len(tok(example["prompt"]).input_ids) for example in dataset)


def load_dpo_dataset(args):
    if str(args.train_file).endswith(".pkl"):
        with open(args.train_file, "rb") as f:
            data = pickle.load(f)
        return Dataset.from_list(data)
    elif str(args.train_file).endswith(".json"):
        return load_dataset("json", data_files=args.train_file, split="train")
    else:
        raise RuntimeError(f"Unknown dataset format: {args.train_file}")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def wait_for_everyone() -> None:
    if dist.is_initialized():
        dist.barrier()


def get_zero_stage(deepspeed_config_path: str | None) -> int:
    if not deepspeed_config_path:
        return 0

    try:
        with open(deepspeed_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        if is_main_process():
            print(
                f"[WARN] Failed to read DeepSpeed config {deepspeed_config_path}: {exc}"
            )
        return 0

    return int(config.get("zero_optimization", {}).get("stage", 0) or 0)


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def release_runtime_memory() -> None:
    gc.collect()
    if is_ascend_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_transformers_deepspeed_context() -> None:
    cleared = False

    try:
        from transformers.integrations.deepspeed import unset_hf_deepspeed_config

        unset_hf_deepspeed_config()
        cleared = True
    except Exception:
        pass

    try:
        import transformers.integrations.deepspeed as ds_integration

        if hasattr(ds_integration, "_hf_deepspeed_config_weak_ref"):
            ds_integration._hf_deepspeed_config_weak_ref = None
            cleared = True
    except Exception:
        pass

    try:
        import transformers.deepspeed as ds_legacy

        if hasattr(ds_legacy, "_hf_deepspeed_config_weak_ref"):
            ds_legacy._hf_deepspeed_config_weak_ref = None
            cleared = True
    except Exception:
        pass

    if cleared:
        print("[INFO] Cleared Transformers DeepSpeed ZeRO-3 context before merge")


def get_post_training_merge_device() -> str:
    if is_ascend_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def zero3_get_peft_model_state_dict(model: PeftModel, adapter_name: str = "default"):
    from deepspeed.utils import safe_get_full_fp32_param
    from peft.utils import PeftType

    config = model.peft_config[adapter_name]
    if config.peft_type != PeftType.LORA:
        raise ValueError("Only LoRA adapters are supported when saving under ZeRO-3.")

    all_params = {name for name, _ in model.named_parameters()}
    bias = config.bias
    if bias == "none":
        candidate_keys = {name for name in all_params if "lora_" in name}
    elif bias == "all":
        candidate_keys = {
            name for name in all_params if "lora_" in name or "bias" in name
        }
    elif bias == "lora_only":
        candidate_keys = {name for name in all_params if "lora_" in name}
        extra_bias = set()
        for name in candidate_keys:
            bias_name = name.split("lora_")[0] + "bias"
            if bias_name in all_params:
                extra_bias.add(bias_name)
        candidate_keys |= extra_bias
    else:
        raise ValueError(f"Unsupported LoRA bias setting: {bias}")

    candidate_keys = {
        name
        for name in candidate_keys
        if (("lora_" in name and adapter_name in name) or ("bias" in name))
    }

    state_dict = {}
    for name, param in model.named_parameters():
        if name not in candidate_keys:
            continue
        full_param = safe_get_full_fp32_param(param)
        if full_param is None:
            continue
        state_dict[name.replace(f".{adapter_name}", "")] = full_param.cpu()
    return state_dict


def zero3_save_lora_model(model: PeftModel, save_directory: str) -> None:
    if os.path.isfile(save_directory):
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )

    for adapter_name, peft_config in model.peft_config.items():
        output_state_dict = zero3_get_peft_model_state_dict(
            model, adapter_name=adapter_name
        )
        output_dir = (
            os.path.join(save_directory, adapter_name)
            if adapter_name != "default"
            else save_directory
        )

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, "adapter_model.bin"))

            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = getattr(
                    model.base_model, "name_or_path", None
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode


def merge_adapter_after_shutdown(
    adapter_output_dir, tokenizer, args, dtype, attn_impl
) -> None:
    merge_device = get_post_training_merge_device()
    merge_dtype = dtype if merge_device != "cpu" else torch.float32
    print(
        f"[INFO] Merging adapter from {adapter_output_dir} into base model on {merge_device} after distributed shutdown"
    )
    clear_transformers_deepspeed_context()
    release_runtime_memory()

    if merge_device.startswith("npu"):
        torch.npu.set_device(merge_device)
    elif merge_device.startswith("cuda"):
        torch.cuda.set_device(merge_device)

    print(f"[INFO] Loading base model for merge with dtype={merge_dtype}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=merge_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if merge_device != "cpu":
        base_model = base_model.to(merge_device)

    print("[INFO] Loading saved adapter for merge")
    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_output_dir,
    )
    print(f"[INFO] Starting merge_and_unload on {merge_device}")
    merged_model = merged_model.merge_and_unload()
    print("[INFO] Merge finished, moving merged model to CPU for saving")
    merged_model = merged_model.to("cpu")
    release_runtime_memory()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] Merged model saved to {args.output_dir}")


def save_model_after_training(trainer, tokenizer, args, dtype, attn_impl) -> str | None:
    model = unwrap_model(trainer.model)
    zero_stage = get_zero_stage(args.deepspeed)
    zero3_enabled = zero_stage == 3

    if zero3_enabled:
        if not isinstance(model, PeftModel):
            raise TypeError("Expected a PeftModel when saving a ZeRO-3 DPO checkpoint.")

        adapter_output_dir = args.output_dir
        if args.save_merged_model:
            adapter_output_dir = os.path.join(args.output_dir, "adapter")

        if is_main_process():
            os.makedirs(adapter_output_dir, exist_ok=True)
            print(f"[INFO] Saving ZeRO-3 LoRA adapter to {adapter_output_dir}")
        zero3_save_lora_model(model, adapter_output_dir)
        wait_for_everyone()

        if not args.save_merged_model and is_main_process():
            tokenizer.save_pretrained(args.output_dir)
        wait_for_everyone()
        return adapter_output_dir if args.save_merged_model else None

    if args.save_merged_model:
        if is_main_process():
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        if is_main_process():
            tokenizer.save_pretrained(args.output_dir)

    wait_for_everyone()
    return None


def main() -> None:

    # Suppress output if current process is not rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        sys.stdout = open(os.devnull, "w")

    # Set device based on local_rank
    if dist.is_initialized():
        local_rank = dist.get_rank()
        set_device(local_rank)

    args = parse_args()
    dataset = load_dpo_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = longest_seq_len(dataset, tokenizer)

    # Initialize a DPO config before loading the model
    # This enables ZeRO stage-3 parameter partitioning
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=max_length,
        beta=args.beta,
        bf16=is_bf16_supported(),
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    # Load model
    # Ascend uses None (CANN auto-handles attention), flash_attn only for CUDA
    attn_impl = None
    if args.flash_attn and not is_ascend_available():
        attn_impl = "flash_attention_2"

    dtype = get_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Print model information
    if dist.get_rank() == 0:
        print(model.device, model)

    # Move model to device
    device_str = get_device_for_rank(dist.get_rank() if dist.is_initialized() else 0)
    model = model.to(device_str)

    # DPOTrainer handles ref_model=None efficiently, especially with PEFT
    ref_model = None

    # Wrap it with PeftModel
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_cfg)

    # Enable gradient checkpointing to save memory
    if args.gradient_checkpointing:
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        print("[INFO] Gradient checkpointing enabled")

    # Create a trainer and train
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    main_process = is_main_process()
    merge_adapter_dir = save_model_after_training(
        trainer, tokenizer, args, dtype, attn_impl
    )

    # Free the training runtime before optional post-processing on rank 0.
    del trainer
    del model
    del ref_model
    release_runtime_memory()

    # Destroy process group
    if dist.is_initialized():
        dist.destroy_process_group()

    if merge_adapter_dir and main_process:
        merge_adapter_after_shutdown(
            merge_adapter_dir,
            tokenizer,
            args,
            dtype,
            attn_impl,
        )


if __name__ == "__main__":
    main()
