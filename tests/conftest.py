"""
Test configuration for the project.
"""
import os
import sys
import pytest

# Add the parent directory to sys.path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
