"""Output formatting for SOUP verification tests."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TextIO

WIDTH = 80
SEP = "=" * WIDTH
THIN_SEP = "-" * WIDTH


@dataclass
class TestResult:
    test_id: str
    requirements: list[str]
    result: str = "PENDING"  # PASS or FAIL
    details: str = ""
    duration_s: float = 0.0
    measured_value: str = ""


def print_test_header(test_id: str, requirements: list[str], description: str) -> None:
    req_str = ", ".join(requirements)
    print(SEP)
    print(f" {test_id} | Requirement: {req_str} | {description}")
    print(SEP)
    print()


def print_step(step_num: int, message: str) -> None:
    print(f"[Step {step_num}] {message}")


def print_detail(message: str) -> None:
    print(f"  {message}")


def print_test_result(result: str, duration_s: float, summary: str) -> None:
    print()
    print(THIN_SEP)
    print(f" RESULT: {result}")
    print(f" Duration: {duration_s:.3f}s")
    print(f" Summary: {summary}")
    print(SEP)
    print()


def print_summary(results: list[TestResult], system_info: dict[str, str]) -> None:
    print()
    print(f"{'=' * 24} VERIFICATION SUMMARY {'=' * 24}")
    print(f" {'Test ID':<18}| {'Requirement(s)':<22}| {'Result':^8}| {'Duration':>10}")
    print(THIN_SEP)
    for r in results:
        req_str = ", ".join(r.requirements)
        print(f" {r.test_id:<18}| {req_str:<22}| {r.result:^8}| {r.duration_s:>9.3f}s")

    passed = sum(1 for r in results if r.result == "PASS")
    total = len(results)
    total_time = sum(r.duration_s for r in results)
    print(THIN_SEP)
    print(f" OVERALL: {passed}/{total} PASSED{' ' * 30}Total: {total_time:.3f}s")
    print()
    for key, value in system_info.items():
        print(f" {key:<20}: {value}")
    print(SEP)


class TeeWriter:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, file: TextIO) -> None:
        self._stdout = sys.stdout
        self._file = file

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()


@contextmanager
def tee_output(file_path: str):
    """Context manager to tee stdout to a file."""
    with open(file_path, "w") as f:
        writer = TeeWriter(f)
        old_stdout = sys.stdout
        sys.stdout = writer
        try:
            yield writer
        finally:
            sys.stdout = old_stdout
