
import torch
import pandas as pd
from model import ImprovedMLP

df = pd.read_csv("data/processed/train_sample.csv")
n_features = df.drop(columns=["Label"]).shape[1]

model = ImprovedMLP(n_features)
model.load_state_dict(torch.load("models/mlp.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, n_features)
torch.onnx.export(model, dummy, "models/mlp.onnx", input_names=["input"], output_names=["output"], opset_version=14)
print("ONNX Exported")
