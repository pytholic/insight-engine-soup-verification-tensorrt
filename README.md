# SOUP Verification — TensorRT 10.13.2.6

Automated SOUP verification test suite for TensorRT 10.13.2.6 per IEC 62304.

## To run on your target machine

1. Pull the docker image with base requirements i.e. `tensorrt-cu12=10.13.2.6` and `cuda-python=12.9.5`
2. Copy `verification/` folder + `image_encoder_WashU.onnx` to the machine
3. `cd verification && pip install -r requirements.txt`
4. `python run_all.py` — builds both engines from ONNX, then runs all 10 tests
5. Screenshot the terminal output for the Word doc
6. Full report also saved to `output/verification_report_<timestamp>.txt`
