"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


sns.set_style("whitegrid")


def plot_glucose_trend(
    df: pd.DataFrame,
    title: str = "Glucose Trend",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """Plot glucose trend dengan threshold lines"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['timestamp'], df['glucose'], label='Glucose', linewidth=1.5)
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Hypo (70)')
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.5, label='Hyper (180)')
    ax.fill_between(df['timestamp'], 70, 180, alpha=0.1, color='green', label='Target range')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Actual",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """Scatter plot prediction vs actual"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # ±20% error bounds
    ax.fill_between([min_val, max_val], 
                    [min_val*0.8, max_val*0.8], 
                    [min_val*1.2, max_val*1.2], 
                    alpha=0.2, color='green', label='±20% error')
    
    ax.set_xlabel('Actual Glucose (mg/dL)')
    ax.set_ylabel('Predicted Glucose (mg/dL)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Clarke Error Grid",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
):
    """Plot Clarke Error Grid"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c='blue')
    
    # Zone boundaries (simplified)
    ax.plot([0, 400], [0, 400], 'k-', lw=1)  # Perfect prediction
    ax.plot([0, 175], [70, 70], 'k--', lw=1)  # Hypo threshold
    ax.plot([70, 70], [84, 400], 'k--', lw=1)
    ax.plot([180, 180], [70, 330], 'k--', lw=1)  # Hyper threshold
    ax.plot([70, 290], [180, 400], 'k--', lw=1)
    
    # Zone labels
    ax.text(30, 370, 'A', fontsize=20, fontweight='bold')
    ax.text(370, 30, 'E', fontsize=20, fontweight='bold')
    ax.text(280, 370, 'B', fontsize=20, fontweight='bold')
    ax.text(160, 370, 'B', fontsize=20, fontweight='bold')
    
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.set_xlabel('Actual Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_multivariate_timeseries(
    df: pd.DataFrame,
    columns: List[str] = ['glucose', 'carbs', 'insulin', 'stress', 'activity'],
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None
):
    """Plot multiple variables over time"""
    n_vars = len(columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    
    for idx, col in enumerate(columns):
        if col in df.columns:
            axes[idx].plot(df['timestamp'], df[col], label=col.capitalize())
            axes[idx].set_ylabel(col.capitalize())
            axes[idx].legend(loc='upper right')
            axes[idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig