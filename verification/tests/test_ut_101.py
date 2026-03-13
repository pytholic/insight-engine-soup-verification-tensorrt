"""SOUP-TRT-UT-101: CUDA Compatibility Test.

Requirement: SOUP-TRT-UR-01
TensorRT shall be compatible with the deployed CUDA Toolkit version.
Acceptance: TensorRT initialization and inference complete without CUDA errors.
"""

from __future__ import annotations

import time

import numpy as np
import tensorrt as trt

from config import ENGINE_PATH, INPUT_DTYPE, INPUT_SHAPE
from cuda_helpers import cuda_initialize, cuda_stream_create, cuda_stream_synchronize
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import (
    allocate_buffers,
    free_buffers,
    get_gpu_info,
    get_tensor_info,
    load_engine,
    run_inference,
)

TEST_ID = "SOUP-TRT-UT-101"
REQUIREMENTS = ["SOUP-TRT-UR-01"]
DESCRIPTION = "CUDA Compatibility Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Initialize CUDA and report versions
    print_step(1, "Initializing CUDA and reporting version information...")
    cuda_initialize()
    print_detail("CUDA runtime initialized successfully (cudaFree(0))")

    gpu_info = get_gpu_info()
    print_detail(f"GPU: {gpu_info['gpu_name']}")
    print_detail(f"CUDA Driver Version: {gpu_info['driver_version']}")
    print_detail(f"CUDA Runtime Version: {gpu_info['cuda_runtime_version']}")
    print_detail(f"Compute Capability: {gpu_info['compute_capability']}")
    print_detail(f"TensorRT Version: {trt.__version__}")
    print()

    # Step 2: Load engine and verify no CUDA errors
    print_step(2, "Loading TensorRT engine...")
    _, engine = load_engine(str(ENGINE_PATH))
    print_detail(f"Engine loaded successfully ({engine.num_io_tensors} I/O tensors)")
    print()

    # Step 3: Run inference
    print_step(3, "Running inference to verify CUDA compatibility...")
    tensors = get_tensor_info(engine)
    input_name = next(t["name"] for t in tensors if t["mode"] == "INPUT")

    context = engine.create_execution_context()
    stream = cuda_stream_create()
    input_memories, output_memories = allocate_buffers(engine)

    input_data = np.random.randn(*INPUT_SHAPE).astype(INPUT_DTYPE)
    outputs = run_inference(context, {input_name: input_data}, input_memories, output_memories, stream)
    print_detail("Inference completed successfully")
    print()

    # Step 4: Synchronize and check for deferred CUDA errors
    print_step(4, "Synchronizing CUDA stream to flush any deferred errors...")
    cuda_stream_synchronize(stream)
    print_detail("CUDA stream synchronized — no deferred errors detected")

    free_buffers(input_memories, output_memories, stream)

    duration = time.perf_counter() - start
    summary = (
        f"CUDA compatibility verified. TensorRT {trt.__version__} initialized and performed "
        f"inference without CUDA errors on {gpu_info['gpu_name']} "
        f"(CUDA {gpu_info['cuda_runtime_version']})."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
