import time
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("models/mlp.onnx")

# Get input shape dynamically
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

n_features = input_shape[1]

print("Model expects features:", n_features)

dummy = np.random.rand(1, n_features).astype("float32")

# CPU Benchmark
cpu_session = ort.InferenceSession(
    "models/mlp.onnx",
    providers=["CPUExecutionProvider"]
)

start = time.time()
for _ in range(200):
    cpu_session.run(None, {input_name: dummy})
end = time.time()

cpu_time = (end - start) / 200 * 1000
print("CPU Avg Time (ms):", round(cpu_time, 3))

# ROCm if available
if "ROCMExecutionProvider" in ort.get_available_providers():
    rocm_session = ort.InferenceSession(
        "models/mlp.onnx",
        providers=["ROCMExecutionProvider"]
    )
    start = time.time()
    for _ in range(200):
        rocm_session.run(None, {input_name: dummy})
    end = time.time()
    rocm_time = (end - start) / 200 * 1000
    print("ROCm Avg Time (ms):", round(rocm_time, 3))