import pandas as pd
from datetime import datetime
import os

def validate_customer_feedback_csv(csv_path: str):
    """Validate customer feedback CSV for data quality issues"""
    FEEDBACK_DATE_COL = 'Feedback Date'
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        issues = []
        
        # Check for missing Customer IDs
        if df['Customer ID'].isnull().any():
            issues.append("Missing Customer IDs found")
            
        # Check for missing or invalid dates
        df[FEEDBACK_DATE_COL] = pd.to_datetime(df[FEEDBACK_DATE_COL], errors='coerce')
        if df[FEEDBACK_DATE_COL].isnull().any():
            issues.append("Invalid or missing Feedback Date entries found")
            
        # Check for missing ratings
        if df['Rating'].isnull().any():
            issues.append("Missing Rating values found")
            
        # Check rating range (should be 1-5)
        invalid_ratings = df[(df['Rating'] < 1) | (df['Rating'] > 5)]['Rating'].count()
        if invalid_ratings > 0:
            issues.append(f"{invalid_ratings} ratings outside valid range (1-5)")
            
        # Check for missing required fields
        required_fields = ['Customer ID', 'Category', 'Resolution Status', 'Sentiment']
        for field in required_fields:
            if df[field].isnull().any():
                issues.append(f"Missing values in required field: {field}")
                
        if issues:
            print("❌ Data Quality Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ CSV validation passed! No issues found.")
            return True
            
    except Exception as e:
        print(f"❌ Error validating CSV: {e}")
        return False

if __name__ == "__main__":
    csv_path = "data/semi_structured/customer_feedback.csv"
    validate_customer_feedback_csv(csv_path)
