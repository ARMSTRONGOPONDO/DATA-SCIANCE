"""
Utility functions for the Kenyan Fraud Detection project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
import joblib
from typing import Dict, List, Tuple, Union, Any

from src.config import OUTPUT_DIR, MODEL_DIR


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    # Make TensorFlow operations deterministic if possible
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass


def save_model(model: Any, model_name: str) -> str:
    """Save a model to disk.
    
    Args:
        model: The model to save
        model_name: Name for the saved model file
    
    Returns:
        str: Path to the saved model file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if model_name.endswith(('.pkl', '.joblib')):
        path = os.path.join(MODEL_DIR, model_name)
        joblib.dump(model, path)
    elif hasattr(model, 'save'):
        # Assume it's a Keras model if it has a 'save' method
        if not model_name.endswith('.keras'):
            model_name += '.keras'
        path = os.path.join(MODEL_DIR, model_name)
        model.save(path)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
    print(f"Model saved to {path}")
    return path


def load_model(model_path: str) -> Any:
    """Load a model from disk.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        The loaded model
    """
    if model_path.endswith(('.pkl', '.joblib')):
        return joblib.load(model_path)
    elif model_path.endswith('.keras'):
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unknown model file extension: {model_path}")


def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray, 
                  model_name: str, save: bool = True) -> plt.Figure:
    """Plot a precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        model_name: Name of the model for the title
        save: Whether to save the plot to disk
    
    Returns:
        matplotlib Figure object
    """
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, where='post', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: {model_name}\nAP={average_precision:0.2f}')
    
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"pr_curve_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {output_path}")
    
    return plt.gcf()


def correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10), 
                      save_as: str = None) -> plt.Figure:
    """Plot a correlation matrix for the dataset.
    
    Args:
        df: Pandas DataFrame
        figsize: Figure size as (width, height)
        save_as: Filename to save the plot (if None, won't save)
    
    Returns:
        matplotlib Figure object
    """
    corr = df.corr()
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    
    if save_as:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_path}")
    
    return f
