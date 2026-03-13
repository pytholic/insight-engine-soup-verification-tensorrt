"""Helper functions for cuda-python interactions."""

from typing import Any

from cuda.bindings import runtime as cudart


class CudaError(Exception):
    """Exception raised for CUDA errors."""

    pass


def check_cuda_error(err: cudart.cudaError_t, operation: str | None = None) -> None:
    """Check CUDA error and raise exception if error occurred."""
    if err != cudart.cudaError_t.cudaSuccess:
        error_str = cudart.cudaGetErrorString(err)
        msg = (
            f"CUDA Error: {error_str} (during {operation})"
            if operation is not None
            else f"CUDA Error: {error_str}"
        )
        raise CudaError(msg)


def cuda_malloc(size: int) -> int:
    """Allocate memory on device."""
    err, ptr = cudart.cudaMalloc(size)
    check_cuda_error(err, f"cudaMalloc({size} bytes)")
    return ptr


def cuda_malloc_host(size: int) -> int:
    """Allocate pinned (page-locked) memory on host."""
    err, ptr = cudart.cudaMallocHost(size)
    check_cuda_error(err, f"cudaMallocHost({size} bytes)")
    return ptr


def cuda_free(ptr: int) -> None:
    """Free memory on device."""
    (err,) = cudart.cudaFree(ptr)
    check_cuda_error(err, "cudaFree")


def cuda_free_host(ptr: int) -> None:
    """Free pinned memory on host."""
    (err,) = cudart.cudaFreeHost(ptr)
    check_cuda_error(err, "cudaFreeHost")


def cuda_memcpy_htod(dst: int, src: Any, size: int, stream: int = 0) -> None:
    """Copy memory from host to device asynchronously."""
    (err,) = cudart.cudaMemcpyAsync(
        dst, src, size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
    )
    check_cuda_error(err, "cudaMemcpyAsync (Host to Device)")


def cuda_memcpy_dtoh(dst: Any, src: int, size: int, stream: int = 0) -> None:
    """Copy memory from device to host asynchronously."""
    (err,) = cudart.cudaMemcpyAsync(
        dst, src, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
    )
    check_cuda_error(err, "cudaMemcpyAsync (Device to Host)")


def cuda_stream_create() -> int:
    """Create a CUDA stream."""
    err, stream = cudart.cudaStreamCreate()
    check_cuda_error(err, "cudaStreamCreate")
    return stream


def cuda_stream_destroy(stream: int) -> None:
    """Destroy a CUDA stream."""
    (err,) = cudart.cudaStreamDestroy(stream)
    check_cuda_error(err, "cudaStreamDestroy")


def cuda_stream_synchronize(stream: int) -> None:
    """Synchronize a CUDA stream (wait for all operations to complete)."""
    (err,) = cudart.cudaStreamSynchronize(stream)
    check_cuda_error(err, "cudaStreamSynchronize")


def cuda_initialize() -> None:
    """Initialize CUDA runtime."""
    (err,) = cudart.cudaFree(0)
    check_cuda_error(err, "CUDA initialization (cudaFree(0))")
