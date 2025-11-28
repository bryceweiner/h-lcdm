"""
Void Distance Computation Utilities
====================================

Efficient computation of pairwise distances for large void catalogs.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def compute_pairwise_distances(
    positions: np.ndarray,
    chunk_size: int = 5000,
    progress_interval: int = 10
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all positions.
    
    Uses chunked computation for large datasets to manage memory efficiently.
    
    Parameters:
        positions: Array of shape (N, 3) with x, y, z coordinates
        chunk_size: Number of positions to process per chunk (default: 5000)
        progress_interval: Log progress every N chunks (default: 10)
        
    Returns:
        Array of shape (N, N) with pairwise distances
    """
    n_points = len(positions)
    
    if n_points == 0:
        return np.array([])
    
    from scipy.spatial.distance import cdist
    
    # For small datasets, compute all at once
    if n_points <= 10000:
        return cdist(positions, positions, metric='euclidean')
    
    # For large datasets, use chunked approach
    logger.info(f"  Large dataset detected ({n_points:,} points), using chunked distance computation...")
    
    distances_chunks = []
    n_chunks = (n_points + chunk_size - 1) // chunk_size
    
    for i in range(0, n_points, chunk_size):
        end_i = min(i + chunk_size, n_points)
        chunk_positions = positions[i:end_i]
        chunk_distances = cdist(chunk_positions, positions, metric='euclidean')
        distances_chunks.append(chunk_distances)
        
        chunk_num = (i // chunk_size) + 1
        if chunk_num % progress_interval == 0:
            logger.info(f"    Processed {end_i:,}/{n_points:,} points ({chunk_num}/{n_chunks} chunks)...")
    
    return np.vstack(distances_chunks)

