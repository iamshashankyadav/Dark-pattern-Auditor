"""Autonomous E-Commerce Compliance & Dark Pattern Auditor."""

import os
import sys

# Ensure Anaconda Library/bin DLLs (Intel MKL, OpenMP) are discovered on Windows
if sys.platform == "win32":
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib_bin):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_bin)
            except Exception:
                pass

# Pre-load PyTorch so its C++ / MKL DLLs initialize cleanly on Windows
try:
    import torch
except Exception:
    pass

# This project is PyTorch-only. Stop transformers from importing TensorFlow/Flax when they
# happen to be installed: it is slow and breaks outright with Keras 3.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__version__ = "0.1.0"
