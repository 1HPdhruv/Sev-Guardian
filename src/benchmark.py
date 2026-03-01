
import time
import numpy as np
from infer_onnx import predict

dummy = np.random.rand(1, 20).flatten().tolist()

start = time.time()
for _ in range(100):
    predict(dummy)
end = time.time()

print("Avg Inference Time (ms):", (end - start)/100 * 1000)
