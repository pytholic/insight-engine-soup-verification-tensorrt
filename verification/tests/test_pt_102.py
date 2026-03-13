"""SOUP-TRT-PT-102: GPU Memory Usage Test.

Requirement: SOUP-TRT-PR-02
GPU memory usage during a single inference shall not exceed the threshold.
Acceptance: Peak GPU memory delta <= 2GB (sample model).
"""

from __future__ import annotations

import time

import numpy as np

from config import ENGINE_PATH, GPU_MEMORY_THRESHOLD_GB, INPUT_DTYPE, INPUT_SHAPE
from cuda_helpers import cuda_stream_create
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import (
    allocate_buffers,
    free_buffers,
    get_gpu_info,
    get_gpu_memory_used_mb,
    get_tensor_info,
    load_engine,
    run_inference,
)

TEST_ID = "SOUP-TRT-PT-102"
REQUIREMENTS = ["SOUP-TRT-PR-02"]
DESCRIPTION = "GPU Memory Usage Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Record baseline GPU memory
    print_step(1, "Recording baseline GPU memory usage...")
    gpu_info = get_gpu_info()
    baseline_mb = get_gpu_memory_used_mb()
    print_detail(f"GPU: {gpu_info['gpu_name']} ({gpu_info['total_memory_mb']:,} MB total)")
    print_detail(f"Baseline GPU memory used: {baseline_mb:,.1f} MB")
    print()

    # Step 2: Load engine and allocate buffers
    print_step(2, "Loading engine and allocating inference buffers...")
    _, engine = load_engine(str(ENGINE_PATH))
    tensors = get_tensor_info(engine)
    input_name = next(t["name"] for t in tensors if t["mode"] == "INPUT")

    context = engine.create_execution_context()
    stream = cuda_stream_create()
    input_memories, output_memories = allocate_buffers(engine)

    after_alloc_mb = get_gpu_memory_used_mb()
    print_detail(f"GPU memory after allocation: {after_alloc_mb:,.1f} MB")
    print_detail(f"Delta from baseline: {after_alloc_mb - baseline_mb:,.1f} MB")
    print()

    # Step 3: Run inference and measure peak memory
    print_step(3, "Running inference and measuring peak GPU memory...")
    input_data = np.random.randn(*INPUT_SHAPE).astype(INPUT_DTYPE)
    run_inference(context, {input_name: input_data}, input_memories, output_memories, stream)

    peak_mb = get_gpu_memory_used_mb()
    delta_mb = peak_mb - baseline_mb
    delta_gb = delta_mb / 1024.0

    print_detail(f"Peak GPU memory during inference: {peak_mb:,.1f} MB")
    print_detail(f"Memory delta (peak - baseline): {delta_mb:,.1f} MB ({delta_gb:.2f} GB)")
    print_detail(f"Threshold: <= {GPU_MEMORY_THRESHOLD_GB} GB")
    print()

    # Step 4: Check threshold
    print_step(4, "Checking against threshold...")
    passed = delta_gb <= GPU_MEMORY_THRESHOLD_GB
    print_detail(
        f"GPU memory delta ({delta_gb:.2f} GB) "
        f"{'<=' if passed else '>'} threshold ({GPU_MEMORY_THRESHOLD_GB} GB)"
    )

    free_buffers(input_memories, output_memories, stream)

    assert passed, (
        f"GPU memory delta {delta_gb:.2f} GB exceeds threshold {GPU_MEMORY_THRESHOLD_GB} GB"
    )

    duration = time.perf_counter() - start
    summary = (
        f"GPU memory delta: {delta_mb:,.1f} MB ({delta_gb:.2f} GB). "
        f"Threshold: {GPU_MEMORY_THRESHOLD_GB} GB. Within limits."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
        measured_value=f"{delta_gb:.2f} GB",
    )
