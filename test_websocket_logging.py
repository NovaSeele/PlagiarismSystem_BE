"""
Test script to verify the WebSocket logging implementation.
This script checks if each module can log properly without causing errors.
"""

import time
import asyncio
import sys
import os

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules with logging functions
from api.routes.websocket import send_log
from modules.bert_module import log_print as bert_log
from modules.lsa_lda_module import log_print as lsa_log
from modules.fasttext_module import log_print as fasttext_log
from modules.plagiarism_main_module import log_print as main_log

# Test the direct send_log function
print("\nTesting direct send_log function...")
send_log("Test", "Testing direct send_log function", "info")
time.sleep(0.5)

# Test each module's log_print function
print("\nTesting BERT module logging...")
bert_log("BERT", "Testing BERT module logging", "info")
time.sleep(0.5)

print("\nTesting LSA/LDA module logging...")
lsa_log("LSA/LDA", "Testing LSA/LDA module logging", "info")
time.sleep(0.5)

print("\nTesting FastText module logging...")
fasttext_log("FastText", "Testing FastText module logging", "info")
time.sleep(0.5)

print("\nTesting Main module logging...")
main_log("Main", "Testing Main module logging", "info")
time.sleep(0.5)

print("\nAll logging tests completed successfully!")
print(
    "If you saw no errors, the WebSocket logging implementation should be working correctly."
)
print("Start the main server and connect a client to see the logs in real-time.")
