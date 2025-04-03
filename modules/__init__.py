"""
MyScienceTools Package

A collection of scientific data processing and visualization tools.
"""

# Import all public objects from submodules
from .lecroy import LeCroyDATA, SpectrumAnalyzer, get_datalist
from .my_plot_style import (
    set_size, 
    reset_to_defaults, 
    enable_minor_ticks, 
    set_science_style, 
    set_presentation_style,
    colors,
    linestyles,
    markers,
    heat_colors,
    diverging_colors
)
from .myInterpolation import *  # Import all public objects from myInterpolation

# Define package version
__version__ = '2.0.0'

# Define public API
__all__ = [
    # LeCroy data processing
    'LeCroyDATA',
    'SpectrumAnalyzer',
    'get_datalist',
    
    # Plot style functions
    'set_size',
    'reset_to_defaults',
    'enable_minor_ticks',
    'set_science_style',
    'set_presentation_style',
    
    # Plot style variables
    'colors',
    'linestyles',
    'markers',
    'heat_colors',
    'diverging_colors',
    
    # Add all public objects from myInterpolation
    # Note: This assumes myInterpolation defines its own __all__ list
    # If not, we should explicitly list the objects we want to expose
    '*'
]

# Print imported modules only when explicitly requested
if __name__ == '__main__':
    print('Imported modules: lecroy, my_plot_style, myInterpolation')