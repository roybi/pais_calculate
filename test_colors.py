#!/usr/bin/env python3
"""
Simple color test for logging
"""

import logging
import sys

# Simple colored formatter
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset color
    }
    
    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        return f"{color}{log_message}{reset}"

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler with colored output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
colored_formatter = ColoredFormatter('%(levelname)s: %(message)s')
console_handler.setFormatter(colored_formatter)
logger.addHandler(console_handler)

# Test all colors
print("Testing colored logging output:")
print("-" * 40)

logger.info("INFO messages appear in GREEN")
logger.warning("WARNING messages appear in YELLOW") 
logger.error("ERROR messages appear in RED")
logger.critical("CRITICAL messages appear in MAGENTA")
logger.debug("DEBUG messages appear in CYAN")

print("-" * 40)
print("Color test complete!")