"""
H-ΛCDM Data Layer
=================

Handles data loading, processing, and management for the Holographic Lambda Model.

Modules:
    loader: Data downloading and caching
    processors: Data processing and analysis
        - base_processor: Base processing functionality
        - cmb_processor: CMB data processing
        - void_processor: Void data processing
"""

from .loader import DataLoader
from .processors.base_processor import BaseDataProcessor
from .processors.cmb_processor import CMBDataProcessor
from .processors.void_processor import VoidDataProcessor

__all__ = [
    'DataLoader',
    'BaseDataProcessor',
    'CMBDataProcessor',
    'VoidDataProcessor'
]
