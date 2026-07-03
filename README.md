# Smart Lender - Applicant Creditworthiness Prediction System

Smart Lender is a machine learning-powered web application designed to evaluate loan applicants' profiles and predict loan eligibility. The platform uses classification algorithms to assist financial institutions in making faster, data-driven decisions.

## Features
- **Exploratory Data Analysis (EDA)**: Interactive distribution and count plots analyzing applicant demographics and core features.
- **Multi-Model ML Pipeline**: Evaluates **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, and **XGBoost** models.
- **Production-Ready Classifier**: Automatically trains, evaluates, and integrates the best-performing model (XGBoost) into the app.
- **Premium Web Dashboard**: A dark-themed web application built with Flask and styled with responsive vanilla CSS.
- **Actionable Scenarios**: Automatically detects and handles low-risk (fast-track) and high-risk applicants.

## Codebase Structure
```
smart-lender/
├── data/
│   └── loan_prediction.csv        # Downloaded dataset
├── models/
│   ├── best_model.pkl             # Serialized XGBoost model
│   └── preprocessor.pkl           # Preprocessor pipeline state
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter notebook for EDA
├── src/
│   ├── app.py                     # Flask web app controller
│   ├── download_data.py           # Script to download dataset
│   ├── test_prediction.py         # HTTP validation script
│   └── train.py                   # Model training and EDA pipeline
├── templates/
│   ├── base.html                  # Core layout structure
│   ├── index.html                 # Dashboard homepage
│   └── predict.html               # Form input & results screen
└── static/
    ├── css/
    │   └── styles.css             # Vanilla CSS design tokens & animations
    └── js/
        └── app.js                 # Form submission loading animations
```

## Setup Instructions

### 1. Initialize Virtual Environment
```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(You can generate `requirements.txt` using `pip freeze > requirements.txt`)*

### 3. Run Pipeline & Start App
```bash
# Train ML models and export the best model
python src/train.py

# Launch the Flask application
python src/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Model Benchmarks
- **Decision Tree**: ~82% Test Accuracy
- **Random Forest**: ~84% Test Accuracy
- **K-Nearest Neighbors**: ~85% Test Accuracy
- **XGBoost**: ~81% Test Accuracy (Saved model)
