"""
Check if PyTorch sees your NVIDIA GPU and CUDA.

Run (with venv activated):
    python check_cuda.py
"""

import torch

print("PyTorch version:", torch.__version__)
print("CUDA built with PyTorch:", torch.version.cuda or "None (CPU-only build)")
print()
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device 0:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
else:
    print()
    print("PyTorch is using a CPU-only build. To use your GTX/RTX GPU:")
    print("  1. Uninstall:  pip uninstall torch")
    print("  2. Reinstall with CUDA:  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("     (or cu118 for CUDA 11.8). See https://pytorch.org/get-started/locally/")
