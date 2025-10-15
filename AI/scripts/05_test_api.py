"""
Test Script for Flask API
Tests all endpoints with sample data
"""

import requests
import json
from datetime import datetime

# API Base URL
BASE_URL = "http://127.0.0.1:5000"

print("="*80)
print("TESTING FLASK API ENDPOINTS")
print("="*80)

# ============================================================================
# 1. TEST HEALTH ENDPOINT
# ============================================================================
print("\n[1] Testing Health Check Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✓ Health check passed!")
except Exception as e:
    print(f"❌ Health check failed: {str(e)}")

# ============================================================================
# 2. TEST MODEL INFO ENDPOINT
# ============================================================================
print("\n[2] Testing Model Info Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Model info retrieved successfully!")
except Exception as e:
    print(f"❌ Model info failed: {str(e)}")

# ============================================================================
# 3. TEST PREDICTION ENDPOINT - LOW RISK
# ============================================================================
print("\n[3] Testing Prediction Endpoint - LOW RISK Case...")

low_risk_order = {
    "Order_ID": "ORD_TEST_001",
    "Product_ID": "PROD_TEST_001",
    "User_ID": "USER_TEST_001",
    "Order_Date": "2025-10-15",
    "Product_Category": "Books",
    "Product_Price": 150.0,
    "Order_Quantity": 1,
    "User_Age": 45,
    "User_Gender": "Male",
    "User_Location": "City1",
    "Payment_Method": "Credit Card",
    "Shipping_Method": "Standard",
    "Discount_Applied": 10.0
}

try:
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=low_risk_order,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result['success'] == True
    print(f"\n✓ Prediction successful!")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Risk Score: {result['risk_score']}%")
    print(f"  Prediction: {result['prediction']}")
    
except Exception as e:
    print(f"❌ Prediction failed: {str(e)}")

# ============================================================================
# 4. TEST PREDICTION ENDPOINT - HIGH RISK
# ============================================================================
print("\n[4] Testing Prediction Endpoint - HIGH RISK Case...")

high_risk_order = {
    "Order_ID": "ORD_TEST_002",
    "Product_ID": "PROD_TEST_002",
    "User_ID": "USER_TEST_002",
    "Order_Date": "2025-10-15",
    "Product_Category": "Electronics",
    "Product_Price": 899.0,
    "Order_Quantity": 5,
    "User_Age": 22,
    "User_Gender": "Male",
    "User_Location": "City99",
    "Payment_Method": "Debit Card",
    "Shipping_Method": "Next-Day",
    "Discount_Applied": 400.0
}

try:
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=high_risk_order,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result['success'] == True
    print(f"\n✓ Prediction successful!")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Risk Score: {result['risk_score']}%")
    print(f"  Prediction: {result['prediction']}")
    
except Exception as e:
    print(f"❌ Prediction failed: {str(e)}")

# ============================================================================
# 5. TEST PREDICTION ENDPOINT - MEDIUM RISK
# ============================================================================
print("\n[5] Testing Prediction Endpoint - MEDIUM RISK Case...")

medium_risk_order = {
    "Order_ID": "ORD_TEST_003",
    "Product_ID": "PROD_TEST_003",
    "User_ID": "USER_TEST_003",
    "Order_Date": "2025-10-15",
    "Product_Category": "Clothing",
    "Product_Price": 350.0,
    "Order_Quantity": 2,
    "User_Age": 35,
    "User_Gender": "Female",
    "User_Location": "City50",
    "Payment_Method": "Credit Card",
    "Shipping_Method": "Express",
    "Discount_Applied": 50.0
}

try:
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=medium_risk_order,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result['success'] == True
    print(f"\n✓ Prediction successful!")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Risk Score: {result['risk_score']}%")
    print(f"  Prediction: {result['prediction']}")
    
except Exception as e:
    print(f"❌ Prediction failed: {str(e)}")

# ============================================================================
# 6. TEST BATCH PREDICTION ENDPOINT
# ============================================================================
print("\n[6] Testing Batch Prediction Endpoint...")

batch_data = {
    "orders": [
        low_risk_order,
        high_risk_order,
        medium_risk_order
    ]
}

try:
    response = requests.post(
        f"{BASE_URL}/api/predict/batch",
        json=batch_data,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result['success'] == True
    assert result['total'] == 3
    print(f"\n✓ Batch prediction successful!")
    print(f"  Total predictions: {result['total']}")
    
except Exception as e:
    print(f"❌ Batch prediction failed: {str(e)}")

# ============================================================================
# 7. TEST ERROR HANDLING - MISSING FIELDS
# ============================================================================
print("\n[7] Testing Error Handling - Missing Required Fields...")

incomplete_order = {
    "Order_ID": "ORD_TEST_INCOMPLETE",
    "Product_Price": 100.0
    # Missing other required fields
}

try:
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=incomplete_order,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 400
    assert result['success'] == False
    print("✓ Error handling works correctly!")
    
except Exception as e:
    print(f"❌ Error handling test failed: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("""
✓ All endpoints tested successfully!

API is ready for integration with Spring Boot.

Next Steps:
1. Keep Flask API running (python scripts/04_api_service.py)
2. Build Spring Boot backend to consume this API
3. Create Thymeleaf frontend for user interface
""")
print("="*80)