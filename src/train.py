import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_training_pipeline():
    # 1. Download/Load Data
    data_path = "data/loan_prediction.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Downloading...")
        from download_data import download_dataset
        download_dataset()
        
    df = pd.read_csv(data_path)
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Drop Loan_ID as it's not predictive
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
        
    # 2. EDA and Visualization
    os.makedirs("static/images", exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Loan_Status', data=df, palette='Set2')
    plt.title('Loan Approval Status Distribution')
    plt.tight_layout()
    plt.savefig('static/images/eda_target.png')
    plt.close()
    
    # Plot 2: Credit History vs Loan Status
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df, palette='Set1')
    plt.title('Credit History vs Loan Status')
    plt.tight_layout()
    plt.savefig('static/images/eda_credit.png')
    plt.close()
    
    # Plot 3: Education vs Loan Status
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Education', hue='Loan_Status', data=df, palette='Set3')
    plt.title('Education vs Loan Status')
    plt.tight_layout()
    plt.savefig('static/images/eda_education.png')
    plt.close()
    
    # Plot 4: ApplicantIncome Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df['ApplicantIncome'], kde=True, color='purple')
    plt.title('Applicant Income Distribution')
    plt.tight_layout()
    plt.savefig('static/images/eda_income.png')
    plt.close()
    
    # 3. Missing Value Imputation
    # Numeric columns
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    # Categorical columns
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    
    # Print missing value counts
    print("\nMissing values before imputation:")
    print(df.isnull().sum())
    
    # Impute numerical features with Mean
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Impute categorical features with Mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    # 4. Encoding Categorical Variables
    # Encode target Loan_Status (Y -> 1, N -> 0)
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Label encode categorical variables
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # Scale features
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Split Dataset (80% Train, 20% Test)
    # Using stratify and a fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 6. Train Models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=7),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', max_depth=6, n_estimators=120, learning_rate=0.15)
    }
    
    trained_models = {}
    model_performance = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        trained_models[name] = model
        model_performance[name] = {
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc
        }
        
        print(f"\n--- {name} ---")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
    # 7. Save Best Model (XGBoost) and Preprocessor
    os.makedirs("models", exist_ok=True)
    
    best_model = trained_models['XGBoost']
    
    # Save the pipeline/artifacts
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump({
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'encoders': encoders,
            'scaler': scaler,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'model_performance': model_performance
        }, f)
        
    print("\nSaved best model and preprocessing artifacts to models/ directory.")

if __name__ == "__main__":
    run_training_pipeline()
