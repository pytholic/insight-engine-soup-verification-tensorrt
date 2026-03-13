"""Build TensorRT engine from ONNX model."""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from config import ENGINE_FP16_PATH, ENGINE_PATH, ONNX_PATH, OUTPUT_DIR


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = False) -> Path:
    """Convert ONNX model to TensorRT serialized engine.

    Args:
        onnx_path: Path to the ONNX model file.
        engine_path: Path to save the serialized engine.
        fp16: Whether to enable FP16 precision.

    Returns:
        Path to the saved engine file.
    """
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    config.clear_flag(trt.BuilderFlag.TF32)

    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(f"  ONNX Parser Error {i}: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model.")

    # Build serialized engine
    print(f"  Building TensorRT engine (FP16={fp16})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"  Engine saved to: {engine_path} ({engine_path.stat().st_size:,} bytes)")
    return engine_path


def build_all_engines() -> None:
    """Build both FP32 and FP16 engines from the ONNX model."""
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  ONNX model: {ONNX_PATH} ({ONNX_PATH.stat().st_size:,} bytes)")
    print()

    print("  [1/2] Building FP32 engine...")
    build_engine(ONNX_PATH, ENGINE_PATH, fp16=False)
    print()

    print("  [2/2] Building FP16 engine...")
    build_engine(ONNX_PATH, ENGINE_FP16_PATH, fp16=True)
    print()

    print("  All engines built successfully.")


if __name__ == "__main__":
    build_all_engines()
