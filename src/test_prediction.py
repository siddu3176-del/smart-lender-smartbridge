import urllib.request
import urllib.parse
import json

def test_lender_app():
    url = "http://127.0.0.1:5000/predict"
    
    # 1. Low-risk applicant (Scenario 1)
    low_risk_data = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': '8000',
        'CoapplicantIncome': '2000',
        'LoanAmount': '150',
        'Loan_Amount_Term': '360',
        'Credit_History': '1.0',
        'Property_Area': 'Semiurban'
    }
    
    # 2. High-risk applicant (Scenario 2)
    high_risk_data = {
        'Gender': 'Female',
        'Married': 'No',
        'Dependents': '0',
        'Education': 'Not Graduate',
        'Self_Employed': 'Yes',
        'ApplicantIncome': '2000',
        'CoapplicantIncome': '0',
        'LoanAmount': '300',
        'Loan_Amount_Term': '360',
        'Credit_History': '0.0',
        'Property_Area': 'Rural'
    }
    
    for name, data in [("Scenario 1: Low-Risk", low_risk_data), ("Scenario 2: High-Risk", high_risk_data)]:
        print(f"\n--- Testing {name} ---")
        encoded_data = urllib.parse.urlencode(data).encode('utf-8')
        req = urllib.request.Request(url, data=encoded_data, method='POST')
        
        try:
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                
                # Check for prediction outcome in returned HTML
                if "Application Approved" in html:
                    print("Result: APPROVED")
                elif "Application Rejected" in html:
                    print("Result: REJECTED")
                else:
                    print("Result Unknown / Page parsed incorrectly")
                    
                # Look for scenarios in HTML
                if "Scenario 1: Fast-Track Approval" in html:
                    print("Scenario matched: Scenario 1 (Fast-Track)")
                elif "Scenario 2: High-Risk Applicant" in html:
                    print("Scenario matched: Scenario 2 (High-Risk)")
                else:
                    print("Scenario matched: Standard / Other")
                    
                # Look for confidence
                for line in html.split('\n'):
                    if "XGBoost Confidence:" in line or "highlight-text" in line:
                        print(f"Details: {line.strip()}")
                        break
        except Exception as e:
            print(f"Error testing {name}: {e}")

if __name__ == "__main__":
    import time
    # Give the server a second to ensure it is up
    time.sleep(1)
    test_lender_app()
