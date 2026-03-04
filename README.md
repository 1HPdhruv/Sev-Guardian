# SEV Guardian 🔐

Hardware-Accelerated AI Threat Intelligence Platform  
Built for AMD Slingshot Hackathon 2026

## Overview

SEV Guardian is an enterprise-ready AI cybersecurity platform that integrates:

- Ensemble Machine Learning (MLP + XGBoost)
- ONNX Runtime optimization
- ROCm GPU acceleration support
- SEV-SNP confidential compute compatibility
- Real-time SOC dashboard

## Dataset

CIC-IDS2017 (Canadian Institute for Cybersecurity)

## Architecture

Traffic → Preprocessing → Ensemble Model → ONNX → ROCm/CPU → FastAPI → Streamlit Dashboard

## Run Locally

```bash
pip install -r requirements.txt
python -m uvicorn src.app:app --reload
python -m streamlit run dashboard/streamlit_app.py
```
## Live Deployment

The SEV Guardian backend API is deployed on Render.

Base URL:
```bash
https://sev-guardian.onrender.com
```
You can explore the API documentation here:
```bash
https://sev-guardian.onrender.com/docs
```
