"""
plotting_utils - Plotting helper utilities.

Provides common plotting operations using matplotlib.
"""

import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Lazy import matplotlib to avoid import errors if not installed
_plt = None

def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def setup_plot_defaults():
    """
    Set up default plotting parameters for consistent styling.
    """
    plt = _get_plt()
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (10, 6),
        'figure.dpi': 100
    })


def save_figure(
    fig,
    filename: str,
    directory: str,
    dpi: int = 150,
    tight: bool = True
) -> str:
    """
    Save a figure to file, creating directory if needed.
    
    Args:
        fig: Matplotlib figure object
        filename: Filename for the saved figure
        directory: Directory to save in
        dpi: Resolution in dots per inch
        tight: If True, use tight layout
        
    Returns:
        Full path to the saved file
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    
    if tight:
        fig.tight_layout()
    
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved figure: {path}")
    
    return path


def create_figure(
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create a new figure with standard styling.
    
    Args:
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        
    Returns:
        Tuple of (figure, axes)
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def add_reference_line(
    ax,
    value: float,
    orientation: str = 'horizontal',
    color: str = 'red',
    linestyle: str = '--',
    label: str = '',
    linewidth: float = 1.5
):
    """
    Add a reference line to an existing plot.
    
    Args:
        ax: Matplotlib axes object
        value: Value for the reference line
        orientation: 'horizontal' or 'vertical'
        color: Line color
        linestyle: Line style
        label: Legend label
        linewidth: Line width
    """
    if orientation == 'horizontal':
        ax.axhline(y=value, color=color, linestyle=linestyle, 
                   label=label, linewidth=linewidth)
    elif orientation == 'vertical':
        ax.axvline(x=value, color=color, linestyle=linestyle,
                   label=label, linewidth=linewidth)
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
