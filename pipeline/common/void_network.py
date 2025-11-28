"""
Void Network Construction and Analysis
=======================================

Single source of truth for constructing void networks and calculating
clustering coefficients.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def construct_void_network_graph(
    positions: np.ndarray,
    linking_length: float,
    progress_interval: int = 5000
) -> Tuple[nx.Graph, int]:
    """
    Construct a network graph from void positions using a linking length.
    
    Parameters:
        positions: Array of shape (N, 3) with x, y, z coordinates
        linking_length: Maximum distance for edge creation (Mpc)
        progress_interval: Log progress every N nodes (default: 5000)
        
    Returns:
        tuple: (graph, n_edges)
            graph: NetworkX Graph object
            n_edges: Number of edges created
    """
    n_voids = len(positions)
    
    if n_voids == 0:
        return nx.Graph(), 0
    
    # Compute pairwise distances
    from pipeline.common.void_distances import compute_pairwise_distances
    distances = compute_pairwise_distances(positions)
    
    logger.info(f"  Constructing network graph...")
    
    # Construct network graph
    G = nx.Graph()
    G.add_nodes_from(range(n_voids))
    
    # Add edges for voids within linking length
    # Optimize: only check upper triangle since distance matrix is symmetric
    edges_added = 0
    for i in range(n_voids):
        # Use vectorized comparison for better performance
        neighbors = np.where((distances[i, :] <= linking_length) & 
                            (distances[i, :] > 0) & 
                            (np.arange(n_voids) > i))[0]
        for j in neighbors:
            G.add_edge(i, j)
            edges_added += 1
        
        if (i + 1) % progress_interval == 0:
            logger.info(f"    Processed {i + 1:,}/{n_voids:,} nodes, {edges_added:,} edges added...")
    
    logger.info(f"  Network constructed: {edges_added:,} edges")
    
    return G, edges_added

def calculate_clustering_coefficients(
    graph: nx.Graph,
    progress_interval: int = 5000
) -> Tuple[float, List[float]]:
    """
    Calculate global and local clustering coefficients for a void network.
    
    Uses NetworkX optimized implementation with manual fallback for robustness.
    
    Parameters:
        graph: NetworkX Graph object
        progress_interval: Log progress every N nodes for manual calculation (default: 5000)
        
    Returns:
        tuple: (global_clustering, local_clustering_list)
            global_clustering: Mean clustering coefficient across all nodes
            local_clustering_list: List of local clustering coefficients for each node
    """
    if graph.number_of_nodes() == 0:
        return 0.0, []
    
    logger.info(f"  Calculating clustering coefficients...")
    
    try:
        # Use NetworkX's built-in clustering coefficient function (much faster)
        clustering_dict = nx.clustering(graph)
        clustering_coefficients = [clustering_dict[node] for node in graph.nodes()]
        
        # Global clustering coefficient (mean of local coefficients)
        global_clustering = np.mean(clustering_coefficients) if clustering_coefficients else 0.0
        
        logger.info(f"  ✓ Clustering coefficients calculated for {len(clustering_coefficients):,} nodes")
        
        return float(global_clustering), clustering_coefficients
        
    except Exception as e:
        logger.warning(f"  NetworkX clustering calculation failed: {e}, falling back to manual calculation...")
        
        # Fallback to manual calculation if NetworkX fails
        clustering_coefficients = []
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        for i, node in enumerate(nodes):
            neighbors = list(graph.neighbors(node))
            k_i = len(neighbors)
            
            if k_i < 2:
                clustering_coefficients.append(0.0)
            else:
                # Count edges between neighbors (optimized: only check pairs once)
                E_i = 0
                neighbor_set = set(neighbors)
                for u in neighbors:
                    for v in neighbors:
                        if u < v and v in neighbor_set and graph.has_edge(u, v):
                            E_i += 1
                
                local_cc = (2.0 * E_i) / (k_i * (k_i - 1))
                clustering_coefficients.append(local_cc)
            
            # Progress logging for large graphs
            if (i + 1) % progress_interval == 0:
                logger.info(f"    Processed {i + 1:,}/{n_nodes:,} nodes...")
        
        global_clustering = np.mean(clustering_coefficients) if clustering_coefficients else 0.0
        
        return float(global_clustering), clustering_coefficients

def compute_network_statistics(
    graph: nx.Graph,
    linking_length: float,
    linking_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive network statistics.
    
    Parameters:
        graph: NetworkX Graph object
        linking_length: Linking length used (Mpc)
        linking_metadata: Optional metadata from linking length calculation
        
    Returns:
        dict: Network statistics including clustering coefficients
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    mean_degree = 2.0 * n_edges / n_nodes if n_nodes > 0 else 0.0
    
    # Calculate clustering coefficients
    global_clustering, local_clustering = calculate_clustering_coefficients(graph)
    
    # Handle NaN values for JSON compatibility
    def safe_float(value):
        """Convert to float, replacing NaN with None for JSON compatibility."""
        result = float(value)
        return result if not np.isnan(result) else None
    
    stats = {
        'clustering_coefficient': safe_float(global_clustering),
        'clustering_std': safe_float(np.std(local_clustering)) if len(local_clustering) > 1 else 0.0,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'mean_degree': safe_float(mean_degree),
        'linking_length': safe_float(linking_length),
        'local_clustering_coefficients': local_clustering,
        'graph': graph  # Store graph for further analysis
    }
    
    # Add linking metadata if provided
    if linking_metadata:
        stats['mean_reff'] = safe_float(linking_metadata.get('mean_radius', 0.0))
        stats['linking_method'] = linking_metadata.get('final_method', 'unknown')
        if 'mean_separation' in linking_metadata:
            stats['mean_separation'] = safe_float(linking_metadata['mean_separation'])
    
    return stats

def build_void_network(
    catalog: pd.DataFrame,
    linking_length: Optional[float] = None,
    linking_method: str = 'robust'
) -> Dict[str, Any]:
    """
    Complete pipeline for building void network from catalog.
    
    This is the main entry point that orchestrates coordinate conversion,
    linking length calculation, network construction, and statistics.
    
    Parameters:
        catalog: DataFrame with void positions (x,y,z or ra,dec,z)
        linking_length: Optional pre-calculated linking length. If None, calculated automatically.
        linking_method: Method for linking length calculation ('robust', 'density', 'radius')
        
    Returns:
        dict: Complete network analysis results
    """
    from pipeline.common.void_coordinates import get_cartesian_positions
    from pipeline.common.void_stats import calculate_robust_linking_length
    
    # Validate input
    if catalog is None or len(catalog) == 0:
        return {'error': 'Empty catalog'}
    
    # Get Cartesian positions
    positions, was_converted = get_cartesian_positions(catalog)
    
    if was_converted:
        logger.info(f"  Converted spherical coordinates to Cartesian")
    else:
        logger.info(f"  Using existing Cartesian coordinates (x, y, z)")
    
    # Calculate linking length if not provided
    if linking_length is None:
        linking_length, linking_meta = calculate_robust_linking_length(
            catalog, method=linking_method
        )
        
        logger.info(f"  Calculated linking length: {linking_length:.2f} Mpc")
        logger.info(f"  Linking method used: {linking_meta.get('final_method', 'unknown')}")
        if linking_meta.get('override_reason'):
            logger.warning(f"  Linking length override reason: {linking_meta.get('override_reason')}")
        if linking_meta.get('mean_separation'):
            logger.info(f"  Mean separation scale: {linking_meta['mean_separation']:.2f} Mpc")
    else:
        linking_meta = None
        logger.info(f"  Using provided linking length: {linking_length:.2f} Mpc")
    
    # Construct network graph
    graph, n_edges = construct_void_network_graph(positions, linking_length)
    
    # Compute statistics
    stats = compute_network_statistics(graph, linking_length, linking_meta)
    
    return stats

