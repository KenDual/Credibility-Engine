"""
Flask API Service for E-commerce Return Fraud Detection
Serves XGBoost model predictions to Spring Boot application
Author: Your Name
Date: 2025-10-15
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Spring Boot

# ============================================================================
# LOAD MODEL & PREPROCESSING ARTIFACTS
# ============================================================================
print("="*80)
print("LOADING MODEL & PREPROCESSING ARTIFACTS...")
print("="*80)

try:
    # Load trained model
    model = joblib.load('models/xgboost_model.joblib')
    logger.info("✓ Model loaded successfully")
    
    # Load scaler
    scaler = joblib.load('models/scaler.joblib')
    logger.info("✓ Scaler loaded successfully")
    
    # Load label encoders
    label_encoders = joblib.load('models/label_encoders.joblib')
    logger.info("✓ Label encoders loaded successfully")
    
    # Load feature metadata
    with open('models/feature_metadata.json', 'r') as f:
        feature_metadata = json.load(f)
    logger.info("✓ Feature metadata loaded successfully")
    
    # Load model metadata
    with open('models/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    logger.info("✓ Model metadata loaded successfully")
    
    print("\n✓ ALL ARTIFACTS LOADED SUCCESSFULLY!")
    print(f"Model: {model_metadata['model_name']}")
    print(f"Version: {model_metadata['version']}")
    print(f"Features: {feature_metadata['total_features']}")
    print(f"Test AUC: {model_metadata['performance_metrics']['test']['auc']:.4f}")
    
except Exception as e:
    logger.error(f"❌ Error loading artifacts: {str(e)}")
    raise

# ============================================================================
# RISK CLASSIFICATION THRESHOLDS
# ============================================================================
LOW_RISK_THRESHOLD = model_metadata['risk_thresholds']['low_risk']
HIGH_RISK_THRESHOLD = model_metadata['risk_thresholds']['high_risk']

def classify_risk(probability):
    """Classify risk level based on probability"""
    if probability < LOW_RISK_THRESHOLD:
        return 'LOW', 'Trusted'
    elif probability < HIGH_RISK_THRESHOLD:
        return 'MEDIUM', 'Uncertain'
    else:
        return 'HIGH', 'Untrustworthy'

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def preprocess_input(data):
    """
    Preprocess input data to match training format
    
    Args:
        data (dict): Input data with order information
        
    Returns:
        pd.DataFrame: Preprocessed features ready for prediction
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Parse Order_Date
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
        # Extract temporal features
        df['Order_Year'] = df['Order_Date'].dt.year
        df['Order_Month'] = df['Order_Date'].dt.month
        df['Order_Day'] = df['Order_Date'].dt.day
        df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek
        df['Order_Quarter'] = df['Order_Date'].dt.quarter
        df['Order_IsWeekend'] = (df['Order_DayOfWeek'] >= 5).astype(int)
        
        # Days since epoch (for consistency with training)
        # Use fixed reference date from training
        reference_date = pd.Timestamp('2023-01-01')
        df['Days_Since_Start'] = (df['Order_Date'] - reference_date).dt.days
        
        # Numerical features
        df['Total_Order_Value'] = df['Product_Price'] * df['Order_Quantity']
        df['Discount_Percentage'] = np.where(
            df['Total_Order_Value'] > 0,
            (df['Discount_Applied'] / df['Total_Order_Value']) * 100,
            0
        )
        df['Price_Per_Item'] = df['Product_Price'] / df['Order_Quantity']
        df['Discount_Per_Item'] = df['Discount_Applied'] / df['Order_Quantity']
        df['Value_To_Discount_Ratio'] = np.where(
            df['Discount_Applied'] > 0,
            df['Total_Order_Value'] / df['Discount_Applied'],
            999
        )
        
        # Price ranges (match training bins)
        df['Price_Range'] = pd.cut(
            df['Product_Price'], 
            bins=[0, 200, 400, 600, 1000], 
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        # Quantity categories
        df['Quantity_Category'] = pd.cut(
            df['Order_Quantity'],
            bins=[0, 1, 3, 5, 10],
            labels=['Single', 'Few', 'Multiple', 'Bulk']
        )
        
        # Age groups
        df['Age_Group'] = pd.cut(
            df['User_Age'],
            bins=[0, 25, 35, 50, 100],
            labels=['Young', 'Adult', 'Middle', 'Senior']
        )
        
        # Risk indicator flags
        df['High_Discount_Flag'] = (df['Discount_Percentage'] > 40).astype(int)
        df['High_Value_Flag'] = (df['Product_Price'] > 630).astype(int)  # 75th percentile from training
        df['High_Quantity_Flag'] = (df['Order_Quantity'] >= 5).astype(int)
        df['Young_User_Flag'] = (df['User_Age'] < 30).astype(int)
        df['Senior_User_Flag'] = (df['User_Age'] >= 60).astype(int)
        df['Premium_Product_Flag'] = (df['Product_Price'] > 700).astype(int)
        df['Low_Price_Flag'] = (df['Product_Price'] < 200).astype(int)
        
        # Encode categorical features
        categorical_features = [
            'Product_Category', 'User_Gender', 'User_Location',
            'Payment_Method', 'Shipping_Method', 
            'Price_Range', 'Quantity_Category', 'Age_Group'
        ]
        
        for col in categorical_features:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).replace('nan', 'Unknown')
                try:
                    df[f'{col}_Encoded'] = le.transform(df[col])
                except ValueError:
                    # If unseen category, use most frequent (0)
                    logger.warning(f"Unseen category in {col}: {df[col].values[0]}, using default")
                    df[f'{col}_Encoded'] = 0
        
        # Select features in correct order
        feature_columns = feature_metadata['feature_columns']
        X = df[feature_columns]
        
        # Scale numerical features
        numerical_features = feature_metadata['numerical_features']
        X[numerical_features] = scaler.transform(X[numerical_features])
        
        return X
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'E-commerce Return Fraud Detection API',
        'version': model_metadata['version'],
        'status': 'running',
        'endpoints': {
            'predict': '/api/predict',
            'health': '/api/health',
            'model_info': '/api/model/info'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Return model metadata"""
    return jsonify({
        'model_name': model_metadata['model_name'],
        'version': model_metadata['version'],
        'trained_date': model_metadata['trained_date'],
        'num_features': feature_metadata['total_features'],
        'performance': {
            'test_auc': model_metadata['performance_metrics']['test']['auc'],
            'test_accuracy': model_metadata['performance_metrics']['test']['accuracy'],
            'test_f1': model_metadata['performance_metrics']['test']['f1']
        },
        'risk_thresholds': model_metadata['risk_thresholds']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expected JSON input:
    {
        "Order_ID": "ORD12345",
        "Product_ID": "PROD67890",
        "User_ID": "USER11111",
        "Order_Date": "2025-10-15",
        "Product_Category": "Electronics",
        "Product_Price": 599.99,
        "Order_Quantity": 1,
        "User_Age": 28,
        "User_Gender": "Male",
        "User_Location": "City1",
        "Payment_Method": "Credit Card",
        "Shipping_Method": "Express",
        "Discount_Applied": 50.0
    }
    
    Returns:
    {
        "success": true,
        "prediction": "Untrustworthy",
        "probability": 0.75,
        "risk_level": "HIGH",
        "risk_score": 75,
        "confidence": "High",
        "timestamp": "2025-10-15T10:30:00.123456"
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = [
            'Product_Category', 'Product_Price', 'Order_Quantity',
            'User_Age', 'User_Gender', 'User_Location',
            'Payment_Method', 'Shipping_Method', 'Discount_Applied',
            'Order_Date'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Make prediction
        probability = model.predict_proba(X)[0][1]  # Probability of return (class 1)
        prediction_class = int(model.predict(X)[0])
        
        # Classify risk
        risk_level, risk_description = classify_risk(probability)
        risk_score = int(probability * 100)
        
        # Determine confidence based on how far from thresholds
        if probability < 0.2 or probability > 0.8:
            confidence = 'High'
        elif probability < 0.3 or probability > 0.7:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Build response
        response = {
            'success': True,
            'order_id': data.get('Order_ID', 'N/A'),
            'prediction': risk_description,
            'prediction_binary': 'Returned' if prediction_class == 1 else 'Not Returned',
            'probability': round(float(probability), 4),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metadata['version']
        }
        
        # Log prediction
        logger.info(f"Prediction made: Order={data.get('Order_ID')}, Risk={risk_level}, Score={risk_score}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON input:
    {
        "orders": [
            { order_data_1 },
            { order_data_2 },
            ...
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions": [ ... ],
        "total": 10,
        "timestamp": "..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'orders' not in data:
            return jsonify({
                'success': False,
                'error': 'No orders data provided'
            }), 400
        
        orders = data['orders']
        predictions = []
        
        for idx, order in enumerate(orders):
            try:
                # Preprocess
                X = preprocess_input(order)
                
                # Predict
                probability = model.predict_proba(X)[0][1]
                prediction_class = int(model.predict(X)[0])
                risk_level, risk_description = classify_risk(probability)
                risk_score = int(probability * 100)
                
                predictions.append({
                    'order_id': order.get('Order_ID', f'Order_{idx+1}'),
                    'prediction': risk_description,
                    'probability': round(float(probability), 4),
                    'risk_level': risk_level,
                    'risk_score': risk_score
                })
                
            except Exception as e:
                predictions.append({
                    'order_id': order.get('Order_ID', f'Order_{idx+1}'),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total': len(predictions),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/predict',
            '/api/predict/batch',
            '/api/health',
            '/api/model/info'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STARTING FLASK API SERVICE")
    print("="*80)
    print(f"\nModel: {model_metadata['model_name']}")
    print(f"Version: {model_metadata['version']}")
    print(f"Test AUC: {model_metadata['performance_metrics']['test']['auc']:.4f}")
    print(f"\nAPI Endpoints:")
    print(f"  - Home: http://127.0.0.1:5000/")
    print(f"  - Predict: http://127.0.0.1:5000/api/predict")
    print(f"  - Batch: http://127.0.0.1:5000/api/predict/batch")
    print(f"  - Health: http://127.0.0.1:5000/api/health")
    print(f"  - Model Info: http://127.0.0.1:5000/api/model/info")
    print("\n" + "="*80)
    print("Press CTRL+C to stop the server")
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Accept connections from any IP
        port=5000,
        debug=False      # Set to False for production
    )