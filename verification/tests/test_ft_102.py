"""SOUP-TRT-FT-102: Inference and Binding Info Test.

Requirements: SOUP-TRT-FR-02, SOUP-TRT-FR-03
The system shall perform inference with valid input tensors and support standard I/O tensor bindings.
Acceptance: Inference completes, output tensor is produced, and all binding info is queryable.
"""

from __future__ import annotations

import time

import numpy as np

from config import ENGINE_PATH, INPUT_DTYPE, INPUT_SHAPE
from cuda_helpers import cuda_stream_create
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import (
    allocate_buffers,
    free_buffers,
    get_tensor_info,
    load_engine,
    run_inference,
)

TEST_ID = "SOUP-TRT-FT-102"
REQUIREMENTS = ["SOUP-TRT-FR-02", "SOUP-TRT-FR-03"]
DESCRIPTION = "Inference and Binding Info Test"


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Load engine
    print_step(1, "Loading TensorRT engine...")
    _, engine = load_engine(str(ENGINE_PATH))
    print_detail(f"Engine loaded: {engine.num_io_tensors} I/O tensors")
    print()

    # Step 2: Query and display tensor binding info
    print_step(2, "Querying tensor binding information...")
    tensors = get_tensor_info(engine)
    print_detail(f"{'Idx':<4} {'Name':<20} {'Mode':<8} {'Shape':<25} {'Dtype':<10}")
    print_detail("-" * 67)
    for t in tensors:
        print_detail(
            f"{t['index']:<4} {t['name']:<20} {t['mode']:<8} "
            f"{str(t['shape']):<25} {t['dtype_str']:<10}"
        )
    print()

    # Step 3: Prepare input data and run inference
    print_step(3, "Preparing input data...")
    input_name = None
    for t in tensors:
        if t["mode"] == "INPUT":
            input_name = t["name"]
            break
    assert input_name is not None, "No input tensor found"

    input_data = np.random.randn(*INPUT_SHAPE).astype(INPUT_DTYPE)
    print_detail(f"Input tensor '{input_name}': shape={INPUT_SHAPE}, dtype={INPUT_DTYPE.__name__}")
    print_detail(f"Input data range: [{input_data.min():.4f}, {input_data.max():.4f}]")
    print()

    print_step(4, "Running inference...")
    context = engine.create_execution_context()
    stream = cuda_stream_create()
    input_memories, output_memories = allocate_buffers(engine)
    print_detail("Execution context created")
    print_detail("CUDA stream created")
    print_detail("I/O buffers allocated (pinned host + device memory)")

    outputs = run_inference(context, {input_name: input_data}, input_memories, output_memories, stream)
    print_detail(f"Inference completed successfully")
    print()

    # Step 5: Verify output
    print_step(5, "Verifying output tensors...")
    for name, output in outputs.items():
        print_detail(f"Output '{name}': shape={output.shape}, dtype={output.dtype}")
        print_detail(f"  Range: [{output.min():.6f}, {output.max():.6f}]")
        print_detail(f"  Mean: {output.mean():.6f}, Std: {output.std():.6f}")
        has_nan = np.isnan(output).any()
        has_inf = np.isinf(output).any()
        print_detail(f"  Contains NaN: {has_nan}, Contains Inf: {has_inf}")
        assert not has_nan, "Output contains NaN values"
        assert not has_inf, "Output contains Inf values"
        assert output.size > 0, "Output tensor is empty"

    free_buffers(input_memories, output_memories, stream)

    duration = time.perf_counter() - start
    output_names = list(outputs.keys())
    summary = (
        f"Inference completed successfully. Output tensors produced: {output_names}. "
        f"All {len(tensors)} tensor bindings queried successfully (name, shape, dtype, mode)."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
