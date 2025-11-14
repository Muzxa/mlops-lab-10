FROM python:3.10

WORKDIR /trainer

RUN pip install mlflow scikit-learn joblib pandas

COPY . .

ENV MLFLOW_TRACKING_PORT="5000"

CMD ["python3", "train.py"]