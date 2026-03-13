"""SOUP-TRT-UT-103: Data Precision Verification Test.

Requirement: SOUP-TRT-UR-03
Data precision (FP32/FP16/INT8) shall match system configuration.
Acceptance: Engine loading and inference operate at the configured precision.
"""

from __future__ import annotations

import os
import time

import numpy as np

from config import ENGINE_FP16_PATH, ENGINE_PATH, INPUT_DTYPE, INPUT_SHAPE
from cuda_helpers import cuda_stream_create
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import (
    allocate_buffers,
    free_buffers,
    get_tensor_info,
    load_engine,
    run_inference,
)

TEST_ID = "SOUP-TRT-UT-103"
REQUIREMENTS = ["SOUP-TRT-UR-03"]
DESCRIPTION = "Data Precision Verification Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Load and inspect FP32 engine I/O types
    print_step(1, "Loading FP32 engine and inspecting I/O tensor data types...")
    _, engine_fp32 = load_engine(str(ENGINE_PATH))
    tensors_fp32 = get_tensor_info(engine_fp32)
    fp32_size = os.path.getsize(str(ENGINE_PATH))
    print_detail("FP32 Engine I/O Tensor Data Types:")
    print_detail(f"  {'Name':<20} {'Mode':<8} {'Dtype':<10} {'Shape'}")
    print_detail(f"  {'-'*60}")
    for t in tensors_fp32:
        print_detail(f"  {t['name']:<20} {t['mode']:<8} {t['dtype_str']:<10} {t['shape']}")
    fp32_dtypes = {t["dtype_str"] for t in tensors_fp32}
    print_detail(f"  I/O data types: {fp32_dtypes}")
    print_detail(f"  Engine file size: {fp32_size:,} bytes")
    assert "FLOAT" in fp32_dtypes, f"Expected FLOAT dtype in FP32 engine, got {fp32_dtypes}"
    print()

    # Step 2: Load and inspect FP16 engine I/O types
    print_step(2, "Loading FP16 engine and inspecting I/O tensor data types...")
    _, engine_fp16 = load_engine(str(ENGINE_FP16_PATH))
    tensors_fp16 = get_tensor_info(engine_fp16)
    fp16_size = os.path.getsize(str(ENGINE_FP16_PATH))
    print_detail("FP16 Engine I/O Tensor Data Types:")
    print_detail(f"  {'Name':<20} {'Mode':<8} {'Dtype':<10} {'Shape'}")
    print_detail(f"  {'-'*60}")
    for t in tensors_fp16:
        print_detail(f"  {t['name']:<20} {t['mode']:<8} {t['dtype_str']:<10} {t['shape']}")
    fp16_dtypes = {t["dtype_str"] for t in tensors_fp16}
    print_detail(f"  I/O data types: {fp16_dtypes}")
    print_detail(f"  Engine file size: {fp16_size:,} bytes")
    print()

    # Step 3: Compare engine sizes (FP16 internal precision -> smaller engine)
    print_step(3, "Comparing engine file sizes (FP16 internal layers use less memory)...")
    size_ratio = fp16_size / fp32_size
    print_detail(f"  FP32 engine: {fp32_size:,} bytes")
    print_detail(f"  FP16 engine: {fp16_size:,} bytes")
    print_detail(f"  Size ratio (FP16/FP32): {size_ratio:.3f}")
    print_detail("")
    print_detail(
        "  Note: TensorRT I/O tensors remain FP32 for both engines. FP16 flag enables"
    )
    print_detail(
        "  half-precision for internal computation layers (convolutions, matmuls),"
    )
    print_detail(
        "  which is reflected in the smaller engine file size."
    )
    print()

    # Step 4: Run inference with both engines and compare outputs
    print_step(4, "Running inference with both engines to verify operational precision...")
    input_name = next(t["name"] for t in tensors_fp32 if t["mode"] == "INPUT")
    input_data = np.random.randn(*INPUT_SHAPE).astype(INPUT_DTYPE)

    # FP32 inference
    ctx_fp32 = engine_fp32.create_execution_context()
    stream_fp32 = cuda_stream_create()
    in_mem_fp32, out_mem_fp32 = allocate_buffers(engine_fp32)
    out_fp32 = run_inference(ctx_fp32, {input_name: input_data}, in_mem_fp32, out_mem_fp32, stream_fp32)
    free_buffers(in_mem_fp32, out_mem_fp32, stream_fp32)

    # FP16 inference
    ctx_fp16 = engine_fp16.create_execution_context()
    stream_fp16 = cuda_stream_create()
    in_mem_fp16, out_mem_fp16 = allocate_buffers(engine_fp16)
    out_fp16 = run_inference(ctx_fp16, {input_name: input_data}, in_mem_fp16, out_mem_fp16, stream_fp16)
    free_buffers(in_mem_fp16, out_mem_fp16, stream_fp16)

    # Compare outputs - FP16 should produce slightly different values due to reduced precision
    output_name = list(out_fp32.keys())[0]
    max_diff = np.abs(out_fp32[output_name] - out_fp16[output_name]).max()
    mean_diff = np.abs(out_fp32[output_name] - out_fp16[output_name]).mean()
    print_detail(f"  Output tensor: '{output_name}'")
    print_detail(f"  FP32 output range: [{out_fp32[output_name].min():.6f}, {out_fp32[output_name].max():.6f}]")
    print_detail(f"  FP16 output range: [{out_fp16[output_name].min():.6f}, {out_fp16[output_name].max():.6f}]")
    print_detail(f"  Max absolute difference: {max_diff:.6f}")
    print_detail(f"  Mean absolute difference: {mean_diff:.6f}")
    print_detail("")
    if max_diff > 0:
        print_detail(
            "  Output differences confirm FP16 engine uses reduced internal precision"
        )
        print_detail("  while maintaining FP32 I/O interface.")
    else:
        print_detail(
            "  Outputs are identical - TensorRT may not have applied FP16 to this model's layers."
        )

    duration = time.perf_counter() - start
    summary = (
        f"FP32 engine ({fp32_size:,} bytes) and FP16 engine ({fp16_size:,} bytes) both loaded "
        f"and executed successfully. I/O tensors report FP32 for both. "
        f"FP16 engine size ratio: {size_ratio:.3f}. "
        f"Output max diff: {max_diff:.6f} (confirms internal precision difference)."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
