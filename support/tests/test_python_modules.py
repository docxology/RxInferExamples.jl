"""
Test script for Python support modules.

Tests all modules in support/src/python/ with real methods.
"""

import os
import shutil
import unittest
import tempfile
import logging
import sys
from datetime import datetime

# Fix path to include support/src/python
python_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python"))
sys.path.insert(0, python_root)

from utils import file_utils
from experiment_logging import logging_utils
from statistics import analysis_utils, validation_utils

print("=" * 60)
print("PYTHON SUPPORT MODULES TEST SUITE")
print("=" * 60)
print()

class TestSupportModules(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.master_test_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.master_test_dir)

    def test_file_utils(self):
        print("Testing file_utils... ", end="", flush=True)
        test_dir = os.path.join(self.master_test_dir, "file_utils")
        os.makedirs(test_dir, exist_ok=True)
        
        # Test ensure_directory
        nested = os.path.join(test_dir, "a", "b", "c")
        result = file_utils.ensure_directory(nested)
        self.assertTrue(os.path.isdir(nested))
        self.assertTrue(os.path.isabs(result))
        
        # Test ensure_parent_directory
        test_file = os.path.join(test_dir, "x", "y", "z.txt")
        result_parent = file_utils.ensure_parent_directory(test_file)
        self.assertTrue(os.path.isdir(os.path.dirname(test_file)))
        self.assertTrue(os.path.isabs(result_parent))
        
        # Test find_files
        with open(os.path.join(nested, "test.txt"), "w") as f:
            f.write("test content")
        found = file_utils.find_files(test_dir, r"\.txt$")
        self.assertGreaterEqual(len(found), 1)
        self.assertTrue(all(os.path.isabs(f) for f in found))
        
        # Test safe_write
        safe_path = os.path.join(test_dir, "safe.txt")
        file_utils.safe_write(safe_path, "safe content")
        self.assertTrue(os.path.isfile(safe_path))
        with open(safe_path, 'r') as f:
            self.assertEqual(f.read(), "safe content")
            
        print("✓ PASS")

    def test_logging_utils(self):
        print("Testing logging_utils... ", end="", flush=True)
        log_dir = os.path.join(self.master_test_dir, "logs")
        
        # Test setup_logger (now with emoji)
        logger = logging_utils.setup_logger(log_dir, "config.json", 1234)
        self.assertIsNotNone(logger)
        
        # Test logging functions
        logger.info("Test info message")
        
        class MockResult:
            def __init__(self, passed, messages):
                self.passed = passed
                self.messages = messages
        
        logging_utils.log_validation(logger, MockResult(True, ["All good"]))
        
        # Check log file was created
        log_path = os.path.join(log_dir, "execution.log")
        self.assertTrue(os.path.isfile(log_path))
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("EXECUTION LOG", content)
            self.assertIn("ℹ️", content)
            
        print("✓ PASS")

    def test_statistics(self):
        print("Testing statistics... ", end="", flush=True)
        import numpy as np
        
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        est = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # RMSE
        rmse = analysis_utils.calculate_rmse(real, est)
        self.assertAlmostEqual(rmse, 0.1, places=2)
        
        # VAF
        vaf = analysis_utils.calculate_vaf(real, est)
        self.assertGreater(vaf, 99)
        
        # Threshold validation
        res = validation_utils.validate_threshold(0.95, 0.9, comparison='>=')
        self.assertTrue(res.passed)
        
        print("✓ PASS")

if __name__ == "__main__":
    unittest.main(verbosity=0)
