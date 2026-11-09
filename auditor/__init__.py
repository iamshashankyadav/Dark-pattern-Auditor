"""Autonomous E-Commerce Compliance & Dark Pattern Auditor."""

import os

# This project is PyTorch-only. Stop transformers from importing TensorFlow/Flax when they
# happen to be installed: it is slow and breaks outright with Keras 3.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__version__ = "0.1.0"
