import mlflow
from sklearn.ensemble import RandomForestClassifier
from loader import get_train_test_split_data
import os
import logging
import warnings

# --- Configuration ---
DATA_PATH = "data/telco_churn.csv"
N_ESTIMATORS = 100
MAX_DEPTH = 10
EXPERIMENT_NAME = "Churn_Prediction_Basic"

# --- Setup Logging ---
# Silence verbose MLflow and Alembic logs
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

def train():
    # 1. Load Data
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_split_data(DATA_PATH)

    # 2. Setup MLflow
    # Insert your code here
 
    
    # Enable Autologging (disable system metrics for cleaner output)
    # This captures params, metrics, model artifacts, and system metrics automatically
    # Insert your code here


if __name__ == "__main__":
    train()
