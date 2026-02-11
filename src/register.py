import mlflow
import os
from mlflow.tracking import MlflowClient

# --- Configuration ---
# In a pipeline, we pass this via environment variable.
# For manual testing, we try to detect the latest run.
RUN_ID = os.getenv("MLFLOW_RUN_ID") 
MODEL_NAME = "ChurnModel"
EXPERIMENT_NAME = "Churn_Prediction_Basic"

def get_latest_run_id():
    try:
        last_run = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        if not last_run.empty:
            return last_run.iloc[0].run_id
    except Exception:
        pass
    return None

def register():
    global RUN_ID
    if not RUN_ID:
        print("MLFLOW_RUN_ID not set. Attempting to auto-detect latest run...")
        RUN_ID = get_latest_run_id()
        
    if not RUN_ID:
        print("Error: Could not find any runs to register. Please run src/train.py first.")
        return

    print(f"Registering model from Run ID: {RUN_ID} as '{MODEL_NAME}'...")
    
    # 1. Register Model
    model_uri = f"runs:/{RUN_ID}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    
    print(f"Model registered. Version: {model_details.version}")
    
    # 2. Transition to Staging
    client = MlflowClient()
    print(f"Transitioning version {model_details.version} to Staging...")
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_details.version,
        stage="Staging"
    )
    print("Transition complete.")

if __name__ == "__main__":
    register()
