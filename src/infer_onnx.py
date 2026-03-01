
import numpy as np
import joblib
import onnxruntime as ort
from src.model import ImprovedMLP

scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb.pkl")
threshold = joblib.load("models/threshold.pkl")

providers = ort.get_available_providers()

if "ROCMExecutionProvider" in providers:
    session = ort.InferenceSession("models/mlp.onnx",
                                   providers=["ROCMExecutionProvider"])
else:
    session = ort.InferenceSession("models/mlp.onnx",
                                   providers=["CPUExecutionProvider"])

def predict(features):
    features = np.array(features).reshape(1, -1).astype(np.float32)
    features = scaler.transform(features)

    nn_result = session.run(None, {"input": features})[0][0][0]
    xgb_result = xgb_model.predict_proba(features)[0][1]

    return float((nn_result + xgb_result)/2)
