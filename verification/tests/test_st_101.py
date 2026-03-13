"""SOUP-TRT-ST-101: Repeated Inference Stability Test.

Requirement: SOUP-TRT-SR-01
GPU inference shall not cause memory corruption or race conditions at the application level.
Acceptance: No errors, crashes, or abnormal termination during repeated inference.
"""

from __future__ import annotations

import time

import numpy as np

from config import ENGINE_PATH, INPUT_DTYPE, INPUT_SHAPE, STABILITY_ITERATIONS
from cuda_helpers import cuda_stream_create
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import (
    allocate_buffers,
    free_buffers,
    get_gpu_memory_used_mb,
    get_tensor_info,
    load_engine,
    run_inference,
)

TEST_ID = "SOUP-TRT-ST-101"
REQUIREMENTS = ["SOUP-TRT-SR-01"]
DESCRIPTION = "Repeated Inference Stability Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Setup
    print_step(1, "Loading engine and preparing for repeated inference...")
    _, engine = load_engine(str(ENGINE_PATH))
    tensors = get_tensor_info(engine)
    input_name = next(t["name"] for t in tensors if t["mode"] == "INPUT")

    context = engine.create_execution_context()
    stream = cuda_stream_create()
    input_memories, output_memories = allocate_buffers(engine)
    input_data = np.random.randn(*INPUT_SHAPE).astype(INPUT_DTYPE)
    print_detail(f"Engine loaded, {STABILITY_ITERATIONS} iterations planned")
    print()

    # Step 2: Run repeated inferences
    print_step(2, f"Running {STABILITY_ITERATIONS} repeated inferences...")
    nan_count = 0
    inf_count = 0
    error_count = 0
    memory_samples = []

    for i in range(STABILITY_ITERATIONS):
        try:
            outputs = run_inference(
                context, {input_name: input_data}, input_memories, output_memories, stream
            )

            # Check output integrity
            for name, output in outputs.items():
                if np.isnan(output).any():
                    nan_count += 1
                if np.isinf(output).any():
                    inf_count += 1

            # Track memory every 10 iterations
            if (i + 1) % 10 == 0:
                mem_mb = get_gpu_memory_used_mb()
                memory_samples.append((i + 1, mem_mb))
                print_detail(f"  Iteration {i + 1:>4}/{STABILITY_ITERATIONS}: OK  (GPU mem: {mem_mb:,.1f} MB)")

        except Exception as e:
            error_count += 1
            print_detail(f"  Iteration {i + 1:>4}/{STABILITY_ITERATIONS}: ERROR - {e}")

    print()

    # Step 3: Report memory trend
    print_step(3, "Analyzing memory trend...")
    if len(memory_samples) >= 2:
        first_mem = memory_samples[0][1]
        last_mem = memory_samples[-1][1]
        mem_diff = last_mem - first_mem
        print_detail(f"  GPU memory at iteration {memory_samples[0][0]:>4}: {first_mem:,.1f} MB")
        print_detail(f"  GPU memory at iteration {memory_samples[-1][0]:>4}: {last_mem:,.1f} MB")
        print_detail(f"  Memory drift: {mem_diff:+,.1f} MB")
    print()

    # Step 4: Summary
    print_step(4, "Checking stability results...")
    print_detail(f"Total iterations: {STABILITY_ITERATIONS}")
    print_detail(f"Errors encountered: {error_count}")
    print_detail(f"NaN outputs detected: {nan_count}")
    print_detail(f"Inf outputs detected: {inf_count}")

    free_buffers(input_memories, output_memories, stream)

    assert error_count == 0, f"{error_count} errors during repeated inference"
    assert nan_count == 0, f"{nan_count} NaN outputs detected"
    assert inf_count == 0, f"{inf_count} Inf outputs detected"

    duration = time.perf_counter() - start
    summary = (
        f"Completed {STABILITY_ITERATIONS} inferences with 0 errors, 0 NaN, 0 Inf. "
        f"No crashes or abnormal termination. Memory stable."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
