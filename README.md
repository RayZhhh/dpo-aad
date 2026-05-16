# DPO-AAD: Fine-Tuning LLMs for Automated Algorithm Design

This repository contains the training code for fine-tuning large language models via **Direct Preference Optimization (DPO)** on algorithm design tasks, as described in our paper [Fine-Tuning Large Language Model for Automated Algorithm Design](https://openreview.net/...) (ICLR 2026).

## Overview

```
dpo-aad/
├── create_data/       # Preference dataset creation via FunSearch/EoH
├── train_cuda/        # DPO training on NVIDIA GPUs (CUDA + DeepSpeed)
└── train_ascend/      # DPO training on Huawei Ascend NPUs
```

---

## Step 1: Create the Preference Dataset

See [`create_data/README.md`](create_data/README.md) for the full data generation pipeline.

---

## Step 2: Training Environment Setup

### CUDA (NVIDIA GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft deepspeed accelerate
```

### Ascend (Huawei NPU)

Ascend training requires the CANN toolkit and the Ascend-adapted version of vLLM. Please install the CANN driver and toolkit matching your hardware first, then:

```bash
pip install torch-npu                     # Ascend PyTorch adapter
pip install vllm-ascend                   # vLLM build for Ascend (version must match vLLM)
pip install transformers trl peft deepspeed accelerate
```

> Refer to the [vllm-ascend documentation](https://github.com/vllm-project/vllm-ascend) for the correct version pairing between `vllm` and `vllm-ascend`.

---

## Step 3: DeepSpeed ZeRO Configuration

Both backends use DeepSpeed ZeRO for distributed training. Choose a stage based on your available GPU/NPU memory:

| Stage | What is sharded | When to use |
|-------|-----------------|-------------|
| **ZeRO-1** | Optimizer states | Ample memory; fastest communication |
| **ZeRO-2** | Optimizer states + gradients | Moderate memory pressure; good default |
| **ZeRO-3** | Optimizer states + gradients + **parameters** | Tight memory; required for large models across many devices |

**Stage 2** (`deepspeed_stage2.json`) offloads optimizer states to CPU and is a safe default for most single-node multi-GPU setups.

**Stage 3** (`deepspeed_stage3_no_offload.json`) shards model parameters across all devices, enabling training of models that would not fit on a single GPU. Note that ZeRO-3 requires `ref_model=None` in the DPO trainer; the reference model logits are computed via the frozen LoRA base weights instead.

Config files are located in:
- `train_cuda/deepspeed_config/`
- `train_ascend/deepspeed_config/`

---

## Step 4: Launch Training

### CUDA — torchrun

```bash
torchrun --nproc_per_node=4 train_cuda/train_dpo.py \
    --train_file        path/to/dataset.pkl \
    --model_name_or_path path/to/model \
    --output_dir        ./output \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs  5 \
    --lora_rank         32 \
    --lora_alpha        32 \
    --deepspeed         train_cuda/deepspeed_config/deepspeed_stage3_no_offload.json \
    --save_merged_model
```

Add `--flash_attn` if your GPU supports Flash Attention 2 (Ampere or newer).

### CUDA — Python API

```python
from train_cuda.trainer import launch_torchrun_dpo

launch_torchrun_dpo(
    model_name_or_path="path/to/model",
    train_file="path/to/dataset.pkl",
    device_ids=[0, 1, 2, 3],
    output_dir="./output",
    per_device_batch_size=1,
    grad_accumulate_steps=4,
    epoch=5,
    lora_rank=32,
    lora_alpha=32,
    deepspeed_config_path="train_cuda/deepspeed_config/deepspeed_stage3_no_offload.json",
    save_merged_model=True,
)
```

### Ascend — torchrun

```bash
torchrun --nproc_per_node=4 train_ascend/train_dpo.py \
    --train_file        path/to/dataset.pkl \
    --model_name_or_path path/to/model \
    --output_dir        ./output \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs  5 \
    --lora_rank         32 \
    --lora_alpha        32 \
    --deepspeed         train_ascend/deepspeed_config/deepspeed_stage3_no_offload.json \
    --save_merged_model
```

### Ascend — Python API

```python
from train_ascend.trainer import launch_torchrun_dpo

launch_torchrun_dpo(
    model_name_or_path="path/to/model",
    train_file="path/to/dataset.pkl",
    device_ids=[0, 1, 2, 3],
    output_dir="./output",
    per_device_batch_size=1,
    grad_accumulate_steps=4,
    epoch=5,
    lora_rank=32,
    lora_alpha=32,
    deepspeed_config_path="train_ascend/deepspeed_config/deepspeed_stage3_no_offload.json",
    save_merged_model=True,
)
```

> **Key difference:** The Ascend backend sets `HCCL_P2P_DISABLE=1` and disables Flash Attention (CANN handles attention kernels automatically). Everything else is interface-compatible with the CUDA backend.

---

## LoRA Configuration

Default target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

Recommended starting point: `lora_rank=32`, `lora_alpha=32`, `lora_dropout=0.05`.
