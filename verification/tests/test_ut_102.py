"""SOUP-TRT-UT-102: GPU Compute Capability Test.

Requirement: SOUP-TRT-UR-02
TensorRT engine shall be built for the target GPU's compute capability.
Acceptance: Engine loads without GPU architecture mismatch errors.
"""

from __future__ import annotations

import time

from config import ENGINE_PATH
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import get_gpu_info, get_tensor_info, load_engine

TEST_ID = "SOUP-TRT-UT-102"
REQUIREMENTS = ["SOUP-TRT-UR-02"]
DESCRIPTION = "GPU Compute Capability Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Query GPU compute capability
    print_step(1, "Querying GPU device properties...")
    gpu_info = get_gpu_info()
    print_detail(f"GPU Name: {gpu_info['gpu_name']}")
    print_detail(f"Compute Capability: {gpu_info['compute_capability']}")
    print_detail(f"Total GPU Memory: {gpu_info['total_memory_mb']:,} MB")
    print()

    # Step 2: Load engine on this GPU
    print_step(2, "Loading TensorRT engine on this GPU...")
    print_detail("If engine was built for a different GPU architecture, loading would fail here.")
    _, engine = load_engine(str(ENGINE_PATH))
    print_detail(f"Engine loaded successfully on {gpu_info['gpu_name']}")
    print_detail(f"Engine has {engine.num_io_tensors} I/O tensors")
    print()

    # Step 3: Verify tensor info is accessible
    print_step(3, "Verifying engine tensor metadata is accessible...")
    tensors = get_tensor_info(engine)
    for t in tensors:
        print_detail(
            f"Tensor '{t['name']}': mode={t['mode']}, shape={t['shape']}, dtype={t['dtype_str']}"
        )

    duration = time.perf_counter() - start
    summary = (
        f"Engine loaded successfully on {gpu_info['gpu_name']} "
        f"(compute capability {gpu_info['compute_capability']}). "
        f"No GPU architecture mismatch errors."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
