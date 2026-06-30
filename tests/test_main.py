import sys
import os
import pandas as pd

# Add src folder to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import load_data, validate_data

def test_automated_validation():
    """Automated test for Requirement (a) and (c)."""
    # 1. Load the data
    df = load_data('/content/train.csv')
    
    # 2. Run automated validation checks
    results = validate_data(df)
    
    # 3. Assertions for CI/CD Feedback
    print(f"Validation Results: {results}")
    
    # Automated Checks
    assert results['missing_values'] == 0, f"Test Failed: Found {results['missing_values']} missing values."
    assert results['duplicates'] == 0, f"Test Failed: Found {results['duplicates']} duplicate rows."
    assert results['positive_values'] == True, "Test Failed: Data contains non-positive values in restricted columns."
    
    print("All automated quality checks passed successfully!")
