
import pandas as pd
import torch
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from model import ImprovedMLP
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/train_sample.csv")
X = df.drop(columns=["Label"]).values.astype(np.float32)
y = df["Label"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "models/xgb.pkl")

# Neural Network
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)

model = ImprovedMLP(X.shape[1])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_f1 = 0
best_threshold = 0.5

for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            probs = torch.sigmoid(model(xb))
            preds.extend(probs.numpy())
            targets.extend(yb.numpy())

    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(targets, (preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            torch.save(model.state_dict(), "models/mlp.pth")

joblib.dump(best_threshold, "models/threshold.pkl")

roc = roc_auc_score(targets, preds)
fpr, tpr, _ = roc_curve(targets, preds)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.savefig("models/roc_curve.png")

cm = confusion_matrix(targets, (preds > best_threshold).astype(int))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")

print("Training complete | Best F1:", best_f1, "| ROC-AUC:", roc)
