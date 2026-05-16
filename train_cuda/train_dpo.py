import os
import argparse
import pickle
import sys
import types


from datasets import load_dataset, Dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel


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
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="Enable gradient checkpointing to save memory")
    return p.parse_args()


def longest_seq_len(dataset, tok):
    # Apply chat template to convert messages to string before tokenizing
    def apply_chat_template(example, key):
        messages = example[key]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

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


def main() -> None:
    # Suppress output if current process is not rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        sys.stdout = open(os.devnull, "w")

    # Set CUDA device based on local_rank
    if dist.is_initialized():
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

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
        bf16=torch.cuda.is_bf16_supported(),
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    # Move model to CUDA
    if dist.is_initialized():
        model = model.to(f"cuda:{dist.get_rank()}")
    else:
        model = model.to("cuda:0")

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
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
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

    # Save merged model
    if args.save_merged_model:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir)
    else:  # Save lora adapter
        trainer.save_model(args.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Destroy process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
