from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
import os
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/flaskdb")
client = MongoClient(mongo_uri)
db = client["flaskdb"]
collection = db["predictions"]

mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
print(f"MLflow tracking URI set to: {mlflow_uri}")

loaded_model = None

def load_from_latest_run():
    """Fallback: Load model from latest experiment run"""
    try:
        mlflow_client = mlflow.MlflowClient()
        experiment = mlflow_client.get_experiment_by_name("Student_Performance_Experiment")
        
        if not experiment:
            print("Experiment 'Student_Performance_Experiment' not found")
            return None
        
        runs = mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            print("No runs found in experiment")
            return None
        
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"Loading model from latest run: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model from run {run_id}")
        return model
        
    except Exception as e:
        print(f"Failed to load model from latest run: {e}")
        return None

def load_latest_registered_model(model_name="student_performance_model"):
    """Load the latest registered model, with fallback to latest run"""
    global loaded_model
    
    if loaded_model is not None:
        return loaded_model
    
    try:
        mlflow_client = mlflow.MlflowClient()
        print(f"Connected to MLflow at {mlflow.get_tracking_uri()}")
    except MlflowException as e:
        print(f"Failed to connect to MLflow server: {e}")
        return None

    try:
        versions = mlflow_client.search_model_versions(f"name='{model_name}'")
        print(f"Found {len(versions)} version(s) for model '{model_name}'")
    except MlflowException as e:
        print(f"Error fetching model versions for '{model_name}': {e}")
        print("Trying to load from latest run instead...")
        return load_from_latest_run()

    if not versions:
        print(f"No registered versions found for model '{model_name}'")
        print("Trying to load from latest run instead...")
        return load_from_latest_run()

    # Pick the latest version
    latest_version = max(versions, key=lambda x: x.creation_timestamp)
    model_uri = f"models:/{model_name}/{latest_version.version}"
    print(f"Loading model: {model_name} version {latest_version.version} from {model_uri}")

    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{model_name}' version {latest_version.version}")
        return loaded_model
    except MlflowException as e:
        print(f"Failed to load registered model: {e}")
        print("Trying to load from latest run instead...")
        return load_from_latest_run()

@app.route('/')
def index():
    predictions = list(collection.find({}, {"_id": 0}).sort("timestamp", -1))
    return render_template('index.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        study_hours = data.get('hours_studied')
        sleep_hours = data.get('hours_slept')

        study_hours = float(study_hours) if study_hours is not None else None
        sleep_hours = float(sleep_hours) if sleep_hours is not None else None

        if study_hours is None or sleep_hours is None:
            return jsonify({"error": "Missing hours_studied or hours_slept"}), 400

        # Load model
        model = load_latest_registered_model()
        if model is None:
            return jsonify({"error": "No model available. Please train a model first."}), 503

        # Prepare input data
        input_data = pd.DataFrame([[study_hours, sleep_hours]], 
                                 columns=['study_hours', 'sleep_hours'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        prediction = int(prediction)
        if isinstance(prediction, (int, float)):
            result = "Pass" if prediction == 1 else "Fail"
        else:
            result = str(prediction)

        # Store prediction in MongoDB
        prediction_doc = {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "result": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        collection.insert_one(prediction_doc)

        return jsonify({
            "prediction": result,
            "study_hours": study_hours,
            "sleep_hours": sleep_hours
        }), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
