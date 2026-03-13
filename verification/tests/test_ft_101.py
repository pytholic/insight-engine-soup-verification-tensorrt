"""SOUP-TRT-FT-101: Engine Loading Test.

Requirement: SOUP-TRT-FR-01
The system shall be able to load a pre-built TensorRT engine file (.engine).
Acceptance: Runtime and engine objects are created without errors.
"""

from __future__ import annotations

import os
import time

import tensorrt as trt

from config import ENGINE_PATH
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result


TEST_ID = "SOUP-TRT-FT-101"
REQUIREMENTS = ["SOUP-TRT-FR-01"]
DESCRIPTION = "Engine Loading Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Check engine file exists
    print_step(1, "Checking engine file exists...")
    engine_path = str(ENGINE_PATH)
    print_detail(f"Path: {engine_path}")
    file_size = os.path.getsize(engine_path)
    print_detail(f"File size: {file_size:,} bytes")
    print_detail("Status: OK")
    print()

    # Step 2: Create TensorRT runtime
    print_step(2, "Creating TensorRT runtime...")
    logger = trt.Logger(trt.Logger.WARNING)
    print_detail("TensorRT Logger initialized (WARNING level)")
    runtime = trt.Runtime(logger)
    assert runtime is not None, "trt.Runtime returned None"
    print_detail(f"trt.Runtime created successfully (type: {type(runtime).__name__})")
    print()

    # Step 3: Deserialize engine
    print_step(3, "Deserializing engine...")
    print_detail("Reading engine bytes...")
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    print_detail(f"Engine bytes read: {len(engine_bytes):,} bytes")
    print_detail("Calling runtime.deserialize_cuda_engine()...")
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    assert engine is not None, "deserialize_cuda_engine returned None"
    print_detail(f"Engine object created: {type(engine).__name__}")
    print_detail(f"Number of I/O tensors: {engine.num_io_tensors}")
    print()

    # Step 4: Verify engine properties
    print_step(4, "Verifying engine properties...")
    num_tensors = engine.num_io_tensors
    assert num_tensors >= 2, f"Expected at least 2 I/O tensors, got {num_tensors}"
    for i in range(num_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print_detail(f"Tensor {i}: name='{name}', mode={mode_str}, shape={shape}, dtype={dtype}")

    duration = time.perf_counter() - start
    summary = (
        f"Engine loaded successfully. Runtime and engine objects created without errors. "
        f"{num_tensors} I/O tensors verified."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
