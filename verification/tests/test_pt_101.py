"""SOUP-TRT-PT-101: Inference Latency Test.

Requirement: SOUP-TRT-PR-01
Inference latency for a reference input shall be within a predefined threshold.
Acceptance: Mean inference time <= 1.0s (sample model).
"""

from __future__ import annotations

import time

import numpy as np

from config import ENGINE_PATH, INPUT_DTYPE, INPUT_SHAPE, LATENCY_THRESHOLD_S
from cuda_helpers import cuda_stream_create
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import allocate_buffers, free_buffers, get_tensor_info, load_engine, run_inference

TEST_ID = "SOUP-TRT-PT-101"
REQUIREMENTS = ["SOUP-TRT-PR-01"]
DESCRIPTION = "Inference Latency Test"

WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 50


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Setup
    print_step(1, "Loading engine and preparing inference pipeline...")
    _, engine = load_engine(str(ENGINE_PATH))
    tensors = get_tensor_info(engine)
    input_name = next(t["name"] for t in tensors if t["mode"] == "INPUT")

    context = engine.create_execution_context()
    stream = cuda_stream_create()
    input_memories, output_memories = allocate_buffers(engine)
    input_data = np.random.randn(*INPUT_SHAPE).astype(INPUT_DTYPE)
    print_detail(f"Engine loaded, buffers allocated, input shape: {INPUT_SHAPE}")
    print()

    # Step 2: Warmup
    print_step(2, f"Running {WARMUP_ITERATIONS} warm-up inferences (not measured)...")
    for i in range(WARMUP_ITERATIONS):
        run_inference(context, {input_name: input_data}, input_memories, output_memories, stream)
    print_detail(f"Warm-up complete ({WARMUP_ITERATIONS} iterations)")
    print()

    # Step 3: Measured inferences
    print_step(3, f"Running {MEASURED_ITERATIONS} measured inferences...")
    latencies = []
    for i in range(MEASURED_ITERATIONS):
        t0 = time.perf_counter()
        run_inference(context, {input_name: input_data}, input_memories, output_memories, stream)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies_arr = np.array(latencies)
    mean_lat = latencies_arr.mean()
    median_lat = np.median(latencies_arr)
    min_lat = latencies_arr.min()
    max_lat = latencies_arr.max()
    p95_lat = np.percentile(latencies_arr, 95)
    p99_lat = np.percentile(latencies_arr, 99)

    print_detail(f"{'Metric':<12} {'Value':>12}")
    print_detail(f"{'-'*24}")
    print_detail(f"{'Mean':<12} {mean_lat:>11.4f}s")
    print_detail(f"{'Median':<12} {median_lat:>11.4f}s")
    print_detail(f"{'Min':<12} {min_lat:>11.4f}s")
    print_detail(f"{'Max':<12} {max_lat:>11.4f}s")
    print_detail(f"{'P95':<12} {p95_lat:>11.4f}s")
    print_detail(f"{'P99':<12} {p99_lat:>11.4f}s")
    print_detail(f"{'Threshold':<12} {'<= ' + str(LATENCY_THRESHOLD_S) + 's':>12}")
    print()

    # Step 4: Check threshold
    print_step(4, "Checking against threshold...")
    passed = mean_lat <= LATENCY_THRESHOLD_S
    print_detail(f"Mean latency ({mean_lat:.4f}s) {'<=' if passed else '>'} threshold ({LATENCY_THRESHOLD_S}s)")

    free_buffers(input_memories, output_memories, stream)

    assert passed, f"Mean latency {mean_lat:.4f}s exceeds threshold {LATENCY_THRESHOLD_S}s"

    duration = time.perf_counter() - start
    summary = (
        f"Mean inference latency: {mean_lat:.4f}s (threshold: {LATENCY_THRESHOLD_S}s). "
        f"Measured over {MEASURED_ITERATIONS} iterations after {WARMUP_ITERATIONS} warm-up."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
        measured_value=f"{mean_lat:.4f}s",
    )
