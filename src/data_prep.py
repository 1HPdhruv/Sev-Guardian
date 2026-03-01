import pandas as pd
from glob import glob
import os

print("Loading CSV files...")
files = glob("data/raw/*.csv")

dfs = []

for f in files:
    print("Reading:", f)
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("Standardizing column names...")

# Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

print("Cleaning dataset...")

# Drop unnecessary columns if present
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Remove inf & NaN
df = df.replace([float("inf"), -float("inf")], pd.NA)
df = df.dropna()

# Check if label column exists
if "Label" not in df.columns:
    print("Available columns:")
    print(df.columns)
    raise Exception("Label column not found!")

# Convert to binary
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

os.makedirs("data/processed", exist_ok=True)

# MacBook-friendly size
df_sample = df.sample(n=150000, random_state=42)

df_sample.to_csv("data/processed/train_sample.csv", index=False)

print("✅ Dataset ready at data/processed/train_sample.csv")