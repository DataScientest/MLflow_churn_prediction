import mlflow
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from agent import create_retention_agent
from dotenv import load_dotenv

load_dotenv(override=True)

def verify_traces():
    """
    Manually triggers a run to verify that traces are correctly captured 
    with the required spans (churn_risk, retention_rules, generate).
    """
    # 1. Setup MLflow
    # Insert your code here

    # 2. Initialize Agent
    # Use the numeric prompt version registered in MLflow 


    # 3. Trigger Trace
    # Scenario: A customer with a specific ID asking for a discount
    test_query = "I am a customer since 3 years (ID: 7590-VHVEG), am I eligible for a discount?"
    
    print(f"Starting trace verification for query: '{test_query}'...")
    
    # Insert your code here


    print("\n--- AGENT OUTPUT ---")
    print(response["output"])
    print("\n--- TRACE VERIFIED ---")
    print("Check MLflow UI (http://localhost:5000) to confirm spans: get_churn_risk, retrieve_retention_rules, generate.")

if __name__ == "__main__":
    verify_traces()
