"""SOUP Verification Test Runner for TensorRT 10.13.2.6.

Usage:
    python run_all.py              # Run all tests (builds engines first)
    python run_all.py --test FT-101  # Run a single test
    python run_all.py --build-only   # Only build engines
    python run_all.py --no-build     # Skip engine build, use existing engines
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Add verification directory to path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent))

import tensorrt as trt

from config import ENGINE_FP16_PATH, ENGINE_PATH, OUTPUT_DIR
from engine_builder import build_all_engines
from formatter import TestResult, print_summary, tee_output
from trt_helpers import get_gpu_info

# Test modules in execution order
TEST_MODULES = [
    ("FT-101", "tests.test_ft_101"),
    ("FT-102", "tests.test_ft_102"),
    ("FT-103", "tests.test_ft_103"),
    ("UT-101", "tests.test_ut_101"),
    ("UT-102", "tests.test_ut_102"),
    ("UT-103", "tests.test_ut_103"),
    ("PT-101", "tests.test_pt_101"),
    ("PT-102", "tests.test_pt_102"),
    ("PT-103", "tests.test_pt_103"),
    ("ST-101", "tests.test_st_101"),
]


def print_header() -> dict[str, str]:
    """Print verification header with system info."""
    print("=" * 80)
    print(" SOUP Verification Test Suite — TensorRT 10.13.2.6")
    print("=" * 80)
    print()

    print(f" Date          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" TensorRT      : {trt.__version__}")

    try:
        gpu_info = get_gpu_info()
        print(f" GPU           : {gpu_info['gpu_name']}")
        print(f" Compute Cap.  : {gpu_info['compute_capability']}")
        print(f" CUDA Runtime  : {gpu_info['cuda_runtime_version']}")
        print(f" CUDA Driver   : {gpu_info['driver_version']}")
        print(f" GPU Memory    : {gpu_info['total_memory_mb']:,} MB")
    except Exception as e:
        gpu_info = {}
        print(f" GPU Info      : Unavailable ({e})")

    print()
    print("=" * 80)
    print()

    system_info = {
        "TensorRT Version": trt.__version__,
        "GPU": gpu_info.get("gpu_name", "N/A"),
        "Compute Capability": gpu_info.get("compute_capability", "N/A"),
        "CUDA Runtime": gpu_info.get("cuda_runtime_version", "N/A"),
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return system_info


def build_engines() -> None:
    """Build TensorRT engines from ONNX model."""
    print("=" * 80)
    print(" BUILDING TENSORRT ENGINES")
    print("=" * 80)
    print()
    build_all_engines()
    print()


def run_single_test(test_id: str, module_name: str) -> TestResult:
    """Run a single test module with error isolation."""
    import importlib

    try:
        module = importlib.import_module(module_name)
        return module.run()
    except AssertionError as e:
        print()
        print(f" RESULT: FAIL")
        print(f" Assertion: {e}")
        print("=" * 80)
        print()
        return TestResult(
            test_id=f"SOUP-TRT-{test_id}",
            requirements=[],
            result="FAIL",
            details=str(e),
        )
    except Exception as e:
        print()
        print(f" RESULT: FAIL (unexpected error)")
        print(f" Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 80)
        print()
        return TestResult(
            test_id=f"SOUP-TRT-{test_id}",
            requirements=[],
            result="FAIL",
            details=f"{type(e).__name__}: {e}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="SOUP TensorRT Verification Tests")
    parser.add_argument("--test", type=str, help="Run single test (e.g., FT-101)")
    parser.add_argument("--build-only", action="store_true", help="Only build engines")
    parser.add_argument("--no-build", action="store_true", help="Skip engine build")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(OUTPUT_DIR / f"verification_report_{timestamp}.txt")

    with tee_output(report_path):
        system_info = print_header()

        # Build engines
        if not args.no_build:
            if ENGINE_PATH.exists() and ENGINE_FP16_PATH.exists() and not args.build_only:
                print("Engines already exist. Use --no-build to skip or delete output/ to rebuild.")
                print(f"  FP32: {ENGINE_PATH} ({ENGINE_PATH.stat().st_size:,} bytes)")
                print(f"  FP16: {ENGINE_FP16_PATH} ({ENGINE_FP16_PATH.stat().st_size:,} bytes)")
                print()
            else:
                build_engines()

        if args.build_only:
            print("Engine build complete. Exiting (--build-only).")
            return 0

        # Determine which tests to run
        if args.test:
            test_list = [(tid, mod) for tid, mod in TEST_MODULES if tid == args.test]
            if not test_list:
                print(f"Unknown test: {args.test}")
                print(f"Available: {', '.join(tid for tid, _ in TEST_MODULES)}")
                return 1
        else:
            test_list = TEST_MODULES

        # Run tests
        results: list[TestResult] = []
        for test_id, module_name in test_list:
            result = run_single_test(test_id, module_name)
            results.append(result)

        # Print summary
        print_summary(results, system_info)

        # Report file location
        print()
        print(f"Report saved to: {report_path}")

    passed = all(r.result == "PASS" for r in results)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
