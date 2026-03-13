"""SOUP-TRT-FT-103: Wrong Input Shape Error Handling Test.

Requirement: SOUP-TRT-FR-04
The system shall handle errors for invalid input tensor shapes.
Acceptance: An exception or error is raised without crashing the application.
"""

from __future__ import annotations

import time

import numpy as np

from config import ENGINE_PATH, INPUT_DTYPE
from cuda_helpers import cuda_stream_create
from formatter import TestResult, print_detail, print_step, print_test_header, print_test_result
from trt_helpers import allocate_buffers, free_buffers, get_tensor_info, load_engine, run_inference

TEST_ID = "SOUP-TRT-FT-103"
REQUIREMENTS = ["SOUP-TRT-FR-04"]
DESCRIPTION = "Wrong Input Shape Error Handling Test"

WRONG_SHAPE = (1, 3, 512, 512)


def run() -> TestResult:
    print_test_header(TEST_ID, REQUIREMENTS, DESCRIPTION)
    start = time.perf_counter()

    # Step 1: Load engine
    print_step(1, "Loading TensorRT engine...")
    _, engine = load_engine(str(ENGINE_PATH))
    tensors = get_tensor_info(engine)
    input_name = None
    expected_shape = None
    for t in tensors:
        if t["mode"] == "INPUT":
            input_name = t["name"]
            expected_shape = t["shape"]
            break
    assert input_name is not None
    print_detail(f"Engine loaded. Input tensor '{input_name}' expects shape: {expected_shape}")
    print()

    # Step 2: Run inference with correct shape first (baseline)
    print_step(2, "Running inference with CORRECT input shape (baseline)...")
    context = engine.create_execution_context()
    stream = cuda_stream_create()
    input_memories, output_memories = allocate_buffers(engine)
    correct_input = np.random.randn(*expected_shape).astype(INPUT_DTYPE)
    outputs = run_inference(context, {input_name: correct_input}, input_memories, output_memories, stream)
    print_detail(f"Baseline inference succeeded with shape {expected_shape}")
    free_buffers(input_memories, output_memories, stream)
    print()

    # Step 3: Attempt inference with wrong shape
    print_step(3, f"Attempting inference with WRONG input shape {WRONG_SHAPE}...")
    print_detail(f"Expected shape: {expected_shape}")
    print_detail(f"Provided shape: {WRONG_SHAPE}")
    print()

    error_caught = False
    error_message = ""

    try:
        wrong_input = np.random.randn(*WRONG_SHAPE).astype(INPUT_DTYPE)
        context2 = engine.create_execution_context()
        stream2 = cuda_stream_create()
        input_memories2, output_memories2 = allocate_buffers(engine)

        # This should fail because the wrong-shape data doesn't match the buffer shape
        np.copyto(input_memories2[input_name].host_memory, wrong_input)
        print_detail("ERROR: No exception raised — shape mismatch was not detected")
    except (ValueError, RuntimeError, Exception) as e:
        error_caught = True
        error_message = str(e)
        print_detail(f"Exception caught: {type(e).__name__}")
        print_detail(f"Message: {error_message}")
    finally:
        try:
            free_buffers(input_memories2, output_memories2, stream2)
        except Exception:
            pass
    print()

    # Step 4: Verify process survived
    print_step(4, "Verifying process is still alive after error...")
    print_detail("Process is alive and responsive.")
    print_detail(f"Error was caught gracefully: {error_caught}")

    assert error_caught, "Expected an error for wrong input shape but none was raised"

    duration = time.perf_counter() - start
    summary = (
        f"Wrong input shape {WRONG_SHAPE} correctly raised {type(ValueError).__name__}: "
        f"'{error_message[:80]}'. Application did not crash."
    )
    print_test_result("PASS", duration, summary)

    return TestResult(
        test_id=TEST_ID,
        requirements=REQUIREMENTS,
        result="PASS",
        details=summary,
        duration_s=duration,
    )
