#!/usr/bin/env python3
"""
Test the logic of the error fix without requiring numpy
"""

def test_error_conditions():
    """Test the conditions that would trigger the error"""
    
    print("Testing error handling logic...")
    
    # Simulate the conditions that cause the error
    test_cases = [
        ("Empty array", {"shape": (0,), "valid": False}),
        ("Single point 1D", {"shape": (1,), "valid": False}),
        ("Single point 2D", {"shape": (1, 1), "valid": False}),
        ("Two points", {"shape": (2, 1), "valid": True}),
        ("Normal data", {"shape": (10, 3), "valid": True}),
        ("Insufficient rows", {"shape": (1, 5), "valid": False}),
        ("Zero columns", {"shape": (5, 0), "valid": False}),
    ]
    
    for name, case in test_cases:
        shape = case["shape"]
        expected_valid = case["valid"]
        
        # Apply our validation logic
        if len(shape) < 2:
            rows, cols = shape[0] if len(shape) > 0 else 0, 0
        else:
            rows, cols = shape[0], shape[1]
        
        # Our validation condition: embedded.shape[0] < 2 or embedded.shape[1] < 1
        is_valid = not (rows < 2 or cols < 1)
        
        status = "PASS" if is_valid == expected_valid else "FAIL"
        print(f"{name:20} shape={shape} valid={is_valid} expected={expected_valid} [{status}]")
    
    print("\nError handling logic test completed!")

if __name__ == "__main__":
    test_error_conditions()