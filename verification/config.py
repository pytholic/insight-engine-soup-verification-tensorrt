"""Configuration constants for SOUP TensorRT verification tests."""

from pathlib import Path

import numpy as np

# Paths
BASE_DIR = Path(__file__).parent
ONNX_PATH = BASE_DIR.parent / "image_encoder_WashU.onnx"
OUTPUT_DIR = BASE_DIR / "output"
ENGINE_PATH = OUTPUT_DIR / "model.engine"
ENGINE_FP16_PATH = OUTPUT_DIR / "model_fp16.engine"

# Model specs
INPUT_NAME = "input"
INPUT_SHAPE = (1, 3, 2048, 1664)
INPUT_DTYPE = np.float32

# Performance thresholds
LATENCY_THRESHOLD_S = 1.0  # PR-01: mean inference latency
GPU_MEMORY_THRESHOLD_GB = 2.0  # PR-02: peak GPU memory delta
ENGINE_LOAD_THRESHOLD_S = 2.0  # PR-03: engine deserialization time

# Stability test
STABILITY_ITERATIONS = 100  # SR-01: number of repeated inferences
