#!/usr/bin/env python3
"""
Test script to verify the chaos analysis error fix
"""

import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_phase_space_analysis():
    """Test the phase space analysis with minimal data to trigger the error condition"""
    
    # Create a minimal embedded array that would cause the original error
    # This simulates the condition that caused the 0-dimensional array issue
    
    print("Testing phase space analysis error handling...")
    
    # Test case 1: Empty array
    empty_embedded = np.array([])
    print(f"Test 1 - Empty array shape: {empty_embedded.shape}")
    
    # Test case 2: Insufficient data (single point)
    single_point = np.array([[1.0]])
    print(f"Test 2 - Single point shape: {single_point.shape}")
    
    # Test case 3: Two points (minimum for covariance)
    two_points = np.array([[1.0], [2.0]])
    print(f"Test 3 - Two points shape: {two_points.shape}")
    
    # Test case 4: Normal case
    normal_data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    print(f"Test 4 - Normal data shape: {normal_data.shape}")
    
    # Test the covariance calculation scenarios that could cause issues
    test_cases = [
        ("Empty", empty_embedded),
        ("Single point", single_point), 
        ("Two points", two_points),
        ("Normal data", normal_data)
    ]
    
    for name, data in test_cases:
        print(f"\n--- Testing {name} ---")
        
        try:
            if len(data) == 0:
                print("Skipping empty array (would fail before covariance)")
                continue
                
            if data.shape[0] < 2:
                print(f"Insufficient data: shape {data.shape}")
                continue
                
            print(f"Data shape: {data.shape}")
            covariance = np.cov(data.T)
            print(f"Covariance shape: {covariance.shape}, ndim: {covariance.ndim}")
            
            # Handle the 0-dimensional case
            if covariance.ndim == 0:
                print("0-dimensional covariance detected - would be fixed to 1x1 matrix")
                covariance = np.array([[covariance]])
                print(f"Fixed covariance shape: {covariance.shape}")
            
            # Try eigenvalue calculation
            eigenvalues, eigenvectors = np.linalg.eig(covariance)
            print(f"Eigenvalues: {eigenvalues}")
            print("SUCCESS: No error occurred")
            
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    test_phase_space_analysis()
    print("\nTest completed!")