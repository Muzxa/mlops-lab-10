#!/bin/bash
echo "Starting MLflow Tracking Server...";
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*"