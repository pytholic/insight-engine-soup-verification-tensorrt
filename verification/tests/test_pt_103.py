"""SOUP-TRT-PT-103: Engine Load Time Test.

Requirement: SOUP-TRT-PR-03
TensorRT engine initial loading time shall be within a reasonable time.
Acceptance: Engine loading time <= 2.0s (sample model).
"""

from __future__ import annotations

import time

import numpy as np
import tensorrt as trt

from config import ENGINE_LOAD_THRESHOLD_S, ENGINE_PATH
from cuda_helpers import cuda_initialize
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result

TEST_ID = "SOUP-TRT-PT-103"
REQUIREMENTS = ["SOUP-TRT-PR-03"]
DESCRIPTION = "Engine Load Time Test"

NUM_ITERATIONS = 5


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Prepare
    print_step(1, "Preparing engine load time measurement...")
    cuda_initialize()
    engine_path = str(ENGINE_PATH)
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    print_detail(f"Engine file: {engine_path}")
    print_detail(f"Engine size: {len(engine_bytes):,} bytes")
    print_detail("Engine bytes pre-loaded into memory (file I/O excluded from measurement)")
    print()

    # Step 2: Measure deserialization time
    print_step(2, f"Measuring engine deserialization time ({NUM_ITERATIONS} iterations)...")
    logger = trt.Logger(trt.Logger.WARNING)
    load_times = []

    for i in range(NUM_ITERATIONS):
        runtime = trt.Runtime(logger)
        t0 = time.perf_counter()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        t1 = time.perf_counter()
        assert engine is not None, f"Deserialization failed on iteration {i}"
        elapsed = t1 - t0
        load_times.append(elapsed)
        print_detail(f"  Iteration {i + 1}: {elapsed:.4f}s")
        del engine

    load_arr = np.array(load_times)
    mean_time = load_arr.mean()
    min_time = load_arr.min()
    max_time = load_arr.max()
    print()

    print_detail(f"{'Metric':<12} {'Value':>12}")
    print_detail(f"{'-'*24}")
    print_detail(f"{'Mean':<12} {mean_time:>11.4f}s")
    print_detail(f"{'Min':<12} {min_time:>11.4f}s")
    print_detail(f"{'Max':<12} {max_time:>11.4f}s")
    print_detail(f"{'Threshold':<12} {'<= ' + str(ENGINE_LOAD_THRESHOLD_S) + 's':>12}")
    print()

    # Step 3: Check threshold
    print_step(3, "Checking against threshold...")
    passed = mean_time <= ENGINE_LOAD_THRESHOLD_S
    print_detail(
        f"Mean load time ({mean_time:.4f}s) "
        f"{'<=' if passed else '>'} threshold ({ENGINE_LOAD_THRESHOLD_S}s)"
    )

    assert passed, (
        f"Mean engine load time {mean_time:.4f}s exceeds threshold {ENGINE_LOAD_THRESHOLD_S}s"
    )

    duration = time.perf_counter() - start
    summary = (
        f"Mean engine deserialization time: {mean_time:.4f}s "
        f"(threshold: {ENGINE_LOAD_THRESHOLD_S}s). "
        f"Measured over {NUM_ITERATIONS} iterations."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
        measured_value=f"{mean_time:.4f}s",
    )
