import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
import joblib
import os

import mlflow
import mlflow.sklearn

PORT = os.getenv("MLFLOW_TRACKING_PORT", "5000")

mlflow.set_tracking_uri(f"http://mlflow:{PORT}")
mlflow.set_experiment("Student_Performance_Experiment")

def train_model(data_path: str, model_path: str):
    # Load dataset
    test_size = 0.2
    random_state = 42
    n_estimators = 100
    data = pd.read_csv(data_path)
    
    # Assume the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Log model with MLflow
    
    with mlflow.start_run() as run:
      mlflow.log_param("n_estimators", n_estimators)
      mlflow.log_param("random_state", random_state)
      mlflow.log_param("test_size", test_size)

      accuracy = model.score(X_test, y_test)
      mlflow.log_metric("accuracy", accuracy)

      signature = infer_signature(X_train, model.predict(X_train))
      mlflow.sklearn.log_model(sk_model=model, name="model", signature=signature, input_example=X_train[:5])

    result = mlflow.register_model(
       f"runs:/{run.info.run_id}/model",
       "student_performance_model"
    )
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model("data/student_performance.csv", "models/random_forest_model.joblib")