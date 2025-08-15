"""
Tests for utility functions.
"""
import os
import numpy as np
import pandas as pd
import pytest
from src.utils import set_random_seed, save_model, load_model

def test_set_random_seed():
    """Test that set_random_seed sets the random seed correctly."""
    set_random_seed(42)
    a = np.random.rand()
    
    set_random_seed(42)
    b = np.random.rand()
    
    assert a == b, "Random seed not working correctly"

def test_save_and_load_model(tmpdir):
    """Test saving and loading a model."""
    # Create a simple model (just a dictionary for testing)
    model = {"weights": np.array([1, 2, 3]), "bias": 0.5}
    
    # Save the model
    model_path = os.path.join(tmpdir, "test_model.pkl")
    save_model(model, model_path)
    
    # Check that the file exists
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    # Load the model
    loaded_model = load_model(model_path)
    
    # Check that the loaded model is the same as the original
    np.testing.assert_array_equal(
        model["weights"], loaded_model["weights"], 
        "Model weights don't match after loading"
    )
    assert model["bias"] == loaded_model["bias"], "Model bias doesn't match after loading"
