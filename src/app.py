import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load model and preprocessor
MODEL_PATH = 'models/best_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        return True
    return False

@app.route('/')
def index():
    artifacts_loaded = load_artifacts()
    performance = None
    if artifacts_loaded and preprocessor:
        performance = preprocessor.get('model_performance', {})
    
    # We will pass the performance metrics of the models to display on the home screen.
    return render_template('index.html', performance=performance)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    artifacts_loaded = load_artifacts()
    if not artifacts_loaded:
        return "Model and preprocessing artifacts not found! Please run the training pipeline first.", 500
        
    if request.method == 'GET':
        return render_template('predict.html')
        
    try:
        # Extract features from form
        gender = request.form.get('Gender', 'Male')
        married = request.form.get('Married', 'No')
        dependents = request.form.get('Dependents', '0')
        education = request.form.get('Education', 'Graduate')
        self_employed = request.form.get('Self_Employed', 'No')
        applicant_income = float(request.form.get('ApplicantIncome', 0))
        coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
        loan_amount = float(request.form.get('LoanAmount', 0))
        loan_amount_term = float(request.form.get('Loan_Amount_Term', 360))
        credit_history = float(request.form.get('Credit_History', 1.0))
        property_area = request.form.get('Property_Area', 'Urban')
        
        # Prepare input dict
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        # Preprocess input data matching the training script logic
        df_input = pd.DataFrame([input_data])
        
        # 1. Fill missing numeric values using the fitted imputer
        num_cols = preprocessor['num_cols']
        df_input[num_cols] = preprocessor['num_imputer'].transform(df_input[num_cols])
        
        # 2. Fill missing categorical values using the fitted imputer
        cat_cols = preprocessor['cat_cols']
        df_input[cat_cols] = preprocessor['cat_imputer'].transform(df_input[cat_cols])
        
        # 3. Label encode categorical columns
        encoders = preprocessor['encoders']
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen labels by mapping them to mode (the first class)
            val = str(df_input.loc[0, col])
            if val not in le.classes_:
                df_input.loc[0, col] = le.classes_[0]
            df_input[col] = le.transform(df_input[col].astype(str))
            
        # 4. Standard scale
        scaler = preprocessor['scaler']
        # The scaler expects all features in the order they were fit
        X_input = df_input[num_cols + cat_cols] # Ensure exact same column ordering
        # Wait, in train.py, the columns of df are ordered as X = df.drop(columns=['Loan_Status'])
        # Let's inspect the order of training features in preprocessor
        # In train.py:
        # num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        # cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        # X = df.drop(columns=['Loan_Status'])
        # So the order of columns in X was:
        # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        # Let's order them exactly as original features:
        feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        X_ordered = df_input[feature_order]
        X_scaled = scaler.transform(X_ordered)
        
        # Predict
        prediction = int(model.predict(X_scaled)[0])
        prob = model.predict_proba(X_scaled)[0]
        confidence = float(prob[prediction] * 100)
        
        result_text = "Approved" if prediction == 1 else "Rejected"
        
        # Scenarios analysis logic
        is_low_risk = (credit_history == 1.0 and applicant_income >= 5000 and prediction == 1)
        is_high_risk = (credit_history == 0.0 or (applicant_income < 3000 and self_employed == 'Yes'))
        
        scenario_tag = "Standard Evaluation"
        if is_low_risk:
            scenario_tag = "Scenario 1: Fast-Track Approval (Low-Risk Applicant)"
        elif is_high_risk:
            scenario_tag = "Scenario 2: High-Risk Applicant Detected"
            
        response_data = {
            'prediction': result_text,
            'confidence': f"{confidence:.1f}%",
            'scenario': scenario_tag,
            'input': input_data
        }
        
        return render_template('predict.html', result=response_data)
        
    except Exception as e:
        import traceback
        print("Error during prediction:")
        traceback.print_exc()
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
