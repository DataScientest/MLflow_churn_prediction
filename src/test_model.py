import requests
import json

# Test the deployed model with correct schema
url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}

# Sample customer data (one-hot encoded as expected by the model)
test_data = {
    "dataframe_split": {
        "columns": [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
            "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
            "MultipleLines_No phone service", "MultipleLines_Yes",
            "InternetService_Fiber optic", "InternetService_No",
            "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
            "OnlineBackup_No internet service", "OnlineBackup_Yes",
            "DeviceProtection_No internet service", "DeviceProtection_Yes",
            "TechSupport_No internet service", "TechSupport_Yes",
            "StreamingTV_No internet service", "StreamingTV_Yes",
            "StreamingMovies_No internet service", "StreamingMovies_Yes",
            "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes",
            "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check"
        ],
        "data": [
            # Example 1: High-risk customer (month-to-month, electronic check, low tenure)
            [0, 2, 70.0, 140.0, True, False, False, True, False, False, True, False, 
             False, False, False, False, False, False, False, False, False, False, 
             False, False, False, False, False, False, True, False],
            
            # Example 2: Low-risk customer (two-year contract, high tenure)
            [0, 60, 65.0, 3900.0, False, True, True, True, False, True, True, False,
             False, True, False, True, False, True, False, True, False, False,
             False, False, False, True, False, True, False, False]
        ]
    }
}

print("Testing model server at:", url)
print("\nSending prediction request...")

try:
    response = requests.post(url, headers=headers, data=json.dumps(test_data))
    response.raise_for_status()
    
    result = response.json()
    print("\n✅ Prediction successful!")
    print(f"Predictions: {result['predictions']}")
    print("\nInterpretation:")
    print(f"  Customer 1 (high-risk): {'Will churn' if result['predictions'][0] == 1 else 'Will NOT churn'}")
    print(f"  Customer 2 (low-risk): {'Will churn' if result['predictions'][1] == 1 else 'Will NOT churn'}")
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ Error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response: {e.response.text}")
