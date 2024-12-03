from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)

# Load the model and preprocessors
print("Loading model and preprocessors...")
model = joblib.load('credit_card_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

# Print model information
print("\nModel type:", type(model).__name__)
print("Model parameters:", model.get_params())

# Print label encoder information
print("\nLabel Encoders classes:")
for col, le in label_encoders.items():
    print(f"\n{col} unique values:", le.classes_)

def convert_date_format(date_str):
    """Convert date from HTML date input format (YYYY-MM-DD) to required format (DD-Mon-YY)"""
    try:
        # Parse the input date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Convert to required format
        return date_obj.strftime('%d-%b-%y')
    except:
        # Return default date if conversion fails
        return "29-Oct-14"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        print("\nInput data:", json.dumps(data, indent=2))
        
        # Convert date format
        if 'Date' in data:
            data['Date'] = convert_date_format(data['Date'])
        else:
            data['Date'] = "29-Oct-14"
        
        print("Converted date:", data['Date'])
        
        # Add dummy Amount (will be removed after scaling)
        data['Amount'] = 0
        
        # Add dummy index
        data['index'] = 0
        
        # Create DataFrame with single row
        df = pd.DataFrame([data])
        print("\nInitial DataFrame:")
        print(df.to_string())
        
        # Ensure all required columns are present
        required_columns = ['index', 'City', 'Date', 'Card Type', 'Exp Type', 'Gender', 'Amount']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col == 'Amount' else ''
        
        # Reorder columns to match training data
        df = df[required_columns]
        print("\nDataFrame after reordering:")
        print(df.to_string())
        
        # Encode categorical variables
        encoded_values = {}
        for col in label_encoders:
            if col in df.columns:
                print(f"\nEncoding {col}:")
                print(f"Original value: {df[col].values[0]}")
                print(f"Available classes: {label_encoders[col].classes_}")
                df[col] = label_encoders[col].transform(df[col].fillna(''))
                encoded_values[col] = df[col].values[0]
                print(f"Encoded value: {df[col].values[0]}")
        
        print("\nAll encoded values:", json.dumps(encoded_values, indent=2))
        
        # Scale features (including Amount)
        df_scaled = scaler.transform(df)
        print("\nScaled features (with column names):")
        for col, val in zip(required_columns, df_scaled[0]):
            print(f"{col}: {val}")
        
        # Remove Amount column after scaling
        df_scaled_no_amount = np.delete(df_scaled, required_columns.index('Amount'), axis=1)
        
        # Try different prediction methods
        if hasattr(model, 'predict_proba'):
            print("\nPrediction probabilities:")
            proba = model.predict_proba(df_scaled_no_amount)
            print(proba)
        
        # Make prediction
        prediction = model.predict(df_scaled_no_amount)[0]
        print("\nFinal Prediction:", prediction)
        
        # If it's a regression model, try to get feature importances
        if hasattr(model, 'feature_importances_'):
            print("\nFeature importances:")
            features = required_columns.copy()
            features.remove('Amount')
            for feature, importance in zip(features, model.feature_importances_):
                print(f"{feature}: {importance}")
        
        return render_template('result.html', 
                             prediction=f"Predicted Amount: ${prediction:.2f}",
                             input_data=data)
    
    except Exception as e:
        import traceback
        error_msg = f"Error in prediction: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # This will show in the Docker logs
        return render_template('result.html', 
                             error=f"Error in prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        
        # Convert date format
        if 'Date' in data:
            data['Date'] = convert_date_format(data['Date'])
        else:
            data['Date'] = "29-Oct-14"
        
        # Add dummy Amount (will be removed after scaling)
        data['Amount'] = 0
        
        # Add dummy index
        data['index'] = 0
        
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        required_columns = ['index', 'City', 'Date', 'Card Type', 'Exp Type', 'Gender', 'Amount']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col == 'Amount' else ''
        
        # Reorder columns to match training data
        df = df[required_columns]
        
        # Encode categorical variables
        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col].fillna(''))
        
        # Scale features (including Amount)
        df_scaled = scaler.transform(df)
        
        # Remove Amount column after scaling
        df_scaled_no_amount = np.delete(df_scaled, required_columns.index('Amount'), axis=1)
        
        # Make prediction
        prediction = model.predict(df_scaled_no_amount)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_amount': float(prediction)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
