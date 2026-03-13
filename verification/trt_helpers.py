"""Shared TensorRT 10.x helper utilities for SOUP verification tests."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
import tensorrt as trt

from cuda_helpers import (
    cuda_free,
    cuda_free_host,
    cuda_initialize,
    cuda_malloc,
    cuda_malloc_host,
    cuda_memcpy_dtoh,
    cuda_memcpy_htod,
    cuda_stream_create,
    cuda_stream_destroy,
    cuda_stream_synchronize,
)

# TensorRT dtype to numpy dtype mapping
TRT_DTYPE_MAP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT64: np.int64,
    trt.DataType.INT8: np.int8,
}


def trt_dtype_to_numpy(dtype: trt.DataType) -> np.dtype:
    """Convert TensorRT data type to numpy dtype."""
    if dtype not in TRT_DTYPE_MAP:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return np.dtype(TRT_DTYPE_MAP[dtype])


def load_engine(engine_path: str) -> tuple[trt.Runtime, trt.ICudaEngine]:
    """Load a serialized TensorRT engine file.

    Returns:
        (runtime, engine) tuple.

    Raises:
        FileNotFoundError: If engine file does not exist.
        RuntimeError: If deserialization fails.
    """
    cuda_initialize()

    with open(engine_path, "rb") as f:
        engine_bytes = f.read()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")

    return runtime, engine


def get_tensor_info(engine: trt.ICudaEngine) -> list[dict]:
    """Query all tensor metadata from engine using TensorRT 10.x API.

    Returns:
        List of dicts with keys: index, name, shape, dtype, mode.
    """
    tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        tensors.append({
            "index": i,
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "dtype_str": str(dtype).split(".")[-1],
            "mode": "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT",
        })
    return tensors


@dataclass
class TensorMemory:
    """Memory allocation for a single tensor."""

    host_memory: np.ndarray
    host_ptr: int
    device_memory: int
    shape: tuple[int, ...]
    nbytes: int


def allocate_buffers(
    engine: trt.ICudaEngine,
) -> tuple[dict[str, TensorMemory], dict[str, TensorMemory]]:
    """Allocate pinned host + device memory for all I/O tensors.

    Returns:
        (input_memories, output_memories) dicts keyed by tensor name.
    """
    input_memories: dict[str, TensorMemory] = {}
    output_memories: dict[str, TensorMemory] = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt_dtype_to_numpy(engine.get_tensor_dtype(name))

        total_elements = int(np.prod(shape, dtype=np.int64))
        nbytes = total_elements * dtype.itemsize

        host_ptr = cuda_malloc_host(nbytes)
        buffer = (ctypes.c_byte * nbytes).from_address(host_ptr)
        host_array = np.frombuffer(buffer, dtype=dtype).reshape(shape)

        device_ptr = cuda_malloc(nbytes)

        memory = TensorMemory(
            host_memory=host_array,
            host_ptr=host_ptr,
            device_memory=device_ptr,
            shape=shape,
            nbytes=nbytes,
        )

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_memories[name] = memory
        else:
            output_memories[name] = memory

    return input_memories, output_memories


def run_inference(
    context: trt.IExecutionContext,
    input_data: dict[str, np.ndarray],
    input_memories: dict[str, TensorMemory],
    output_memories: dict[str, TensorMemory],
    stream: int,
) -> dict[str, np.ndarray]:
    """Run inference using TensorRT 10.x API.

    Args:
        context: TensorRT execution context.
        input_data: Dict of input tensor name -> numpy array.
        input_memories: Allocated input memory buffers.
        output_memories: Allocated output memory buffers.
        stream: CUDA stream handle.

    Returns:
        Dict of output tensor name -> numpy array with results.
    """
    # Copy input data to pinned host memory, then H2D transfer
    for name, data in input_data.items():
        np.copyto(input_memories[name].host_memory, data)
        cuda_memcpy_htod(
            input_memories[name].device_memory,
            input_memories[name].host_memory.ctypes.data,
            input_memories[name].nbytes,
            stream,
        )
        context.set_tensor_address(name, input_memories[name].device_memory)

    # Set output tensor addresses
    for name, mem in output_memories.items():
        context.set_tensor_address(name, mem.device_memory)

    # Execute inference
    success = context.execute_async_v3(stream)
    if not success:
        raise RuntimeError("TensorRT inference failed (execute_async_v3 returned False)")

    # D2H transfer for outputs
    for mem in output_memories.values():
        cuda_memcpy_dtoh(
            mem.host_memory.ctypes.data,
            mem.device_memory,
            mem.nbytes,
            stream,
        )

    cuda_stream_synchronize(stream)

    # Return copies of output arrays
    results: dict[str, np.ndarray] = {}
    for name, mem in output_memories.items():
        results[name] = mem.host_memory.copy().reshape(mem.shape)
    return results


def free_buffers(
    input_memories: dict[str, TensorMemory],
    output_memories: dict[str, TensorMemory],
    stream: int | None = None,
) -> None:
    """Free all allocated memory buffers."""
    for mem in list(input_memories.values()) + list(output_memories.values()):
        try:
            cuda_free(mem.device_memory)
        except Exception:
            pass
        try:
            cuda_free_host(mem.host_ptr)
        except Exception:
            pass
    if stream is not None:
        try:
            cuda_stream_destroy(stream)
        except Exception:
            pass


def get_gpu_info() -> dict[str, str]:
    """Get GPU device info via CUDA runtime."""
    from cuda.bindings import runtime as cudart

    err, props = cudart.cudaGetDeviceProperties(0)
    name = props.name
    if isinstance(name, bytes):
        name = name.decode("utf-8").rstrip("\x00")

    cc_major = props.major
    cc_minor = props.minor

    err, driver_version = cudart.cudaDriverGetVersion()
    driver_major = driver_version // 1000
    driver_minor = (driver_version % 1000) // 10

    err, runtime_version = cudart.cudaRuntimeGetVersion()
    runtime_major = runtime_version // 1000
    runtime_minor = (runtime_version % 1000) // 10

    return {
        "gpu_name": name,
        "compute_capability": f"{cc_major}.{cc_minor}",
        "driver_version": f"{driver_major}.{driver_minor}",
        "cuda_runtime_version": f"{runtime_major}.{runtime_minor}",
        "total_memory_mb": props.totalGlobalMem // (1024 * 1024),
    }


def get_gpu_memory_used_mb() -> float:
    """Get current GPU memory usage in MB via pynvml."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mb = mem_info.used / (1024 * 1024)
    pynvml.nvmlShutdown()
    return used_mb
