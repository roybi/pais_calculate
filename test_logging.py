#!/usr/bin/env python3
"""
Simple test script to verify logging functionality
"""

import logging
from pathlib import Path
from datetime import datetime

# Enhanced logging setup
def setup_detailed_logging():
    """Setup comprehensive logging with file and console output"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with detailed logging
    file_handler = logging.FileHandler(
        log_dir / f"test_logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Setup logging
logger = setup_detailed_logging()

def log_function_entry(func_name: str, **kwargs):
    """Log function entry with parameters"""
    params_str = ', '.join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
    logger.info(f"[ENTER] {func_name}({params_str})")

def log_function_exit(func_name: str, success: bool = True, result_info: str = ""):
    """Log function exit with result info"""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"[EXIT] {func_name} - {status} {result_info}")

def log_function_error(func_name: str, error: Exception):
    """Log function error with details"""
    logger.error(f"[ERROR] {func_name}: {str(error)}", exc_info=True)

def test_function():
    """Test function to demonstrate logging"""
    log_function_entry("test_function", param1="value1", param2=42)
    
    try:
        logger.info("Performing some test operations...")
        
        # Simulate some work
        for i in range(3):
            logger.info(f"Processing step {i+1}/3")
        
        # Simulate success
        log_function_exit("test_function", success=True, result_info="All operations completed")
        return True
        
    except Exception as e:
        log_function_error("test_function", e)
        log_function_exit("test_function", success=False, result_info=f"Failed: {str(e)}")
        return False

def test_function_with_error():
    """Test function to demonstrate error logging"""
    log_function_entry("test_function_with_error")
    
    try:
        logger.info("About to trigger an error...")
        raise ValueError("This is a test error")
        
    except Exception as e:
        log_function_error("test_function_with_error", e)
        log_function_exit("test_function_with_error", success=False, result_info=f"Failed as expected: {str(e)}")
        return False

def main():
    """Main test function"""
    log_function_entry("main")
    
    try:
        logger.info("Starting logging functionality test")
        
        # Test successful function
        logger.info("Testing successful function execution...")
        success1 = test_function()
        
        # Test function with error
        logger.info("Testing function with error handling...")
        success2 = test_function_with_error()
        
        logger.info("Logging test completed successfully!")
        log_function_exit("main", success=True, result_info="All logging tests passed")
        
        return 0
        
    except Exception as e:
        log_function_error("main", e)
        log_function_exit("main", success=False, result_info=f"Test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())