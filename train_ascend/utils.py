"""Device abstraction utilities for Ascend/CUDA compatibility."""

import torch


def is_ascend_available():
    """Check if Ascend NPU is available."""
    try:
        import torch_npu
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def get_device_name():
    """Get the device type: 'npu' or 'cuda'."""
    return "npu" if is_ascend_available() else "cuda"


def set_device(local_rank):
    """Set the current device."""
    if is_ascend_available():
        torch.npu.set_device(f"npu:{local_rank}")
    else:
        torch.cuda.set_device(local_rank)


def is_bf16_supported():
    """Check if BF16 is supported."""
    if is_ascend_available():
        return torch.npu.is_bf16_supported()
    return torch.cuda.is_bf16_supported()


def get_device_for_rank(rank):
    """Get device string for a given rank."""
    return f"{get_device_name()}:{rank}"


def get_dtype():
    """Get the appropriate dtype based on device support."""
    if is_ascend_available():
        return torch.bfloat16 if torch.npu.is_bf16_supported() else torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
