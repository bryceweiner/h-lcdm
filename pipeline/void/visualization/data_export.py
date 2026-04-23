"""
Void Visualization Data Export
==============================

Export void catalog data and network structure for 3D visualization.

This module extracts void positions, properties, and network connectivity
from the processed void catalog and clustering analysis results.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def export_void_visualization_data(
    void_catalog_path: str = "processed_data/voids_deduplicated.pkl",
    results_path: str = "results/json/void_results.json",
    output_path: str = "results/figures/void/void_map_data.json",
    results_dict: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export void data for 3D visualization.

    Extracts void positions, radii, orientations, clustering coefficients,
    and network edges from processed void catalog and analysis results.

    Parameters:
        void_catalog_path: Path to processed void catalog pickle file
        results_path: Path to void analysis results JSON file
        output_path: Path to output JSON file for 3D visualization
        results_dict: Optional results dictionary (if provided, results_path is ignored)

    Returns:
        str: Path to exported JSON file
    """
    logger.info("Exporting void visualization data...")

    # Load void catalog
    try:
        catalog_df = pd.read_pickle(void_catalog_path)
        logger.info(f"Loaded void catalog with {len(catalog_df)} voids")
    except Exception as e:
        raise FileNotFoundError(f"Could not load void catalog: {e}")

    # Load analysis results (either from dict or file)
    if results_dict is not None:
        results = results_dict
        logger.info("Using provided results dictionary")
    else:
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            logger.info("Loaded void analysis results")
        except Exception as e:
            raise FileNotFoundError(f"Could not load results: {e}")

    # Extract void data
    voids_data = _extract_void_data(catalog_df, results)

    # Extract network edges
    edges_data = _extract_network_edges(catalog_df, results)

    # Create metadata
    metadata = _create_metadata(catalog_df, results, voids_data, edges_data)

    # Combine into final data structure
    visualization_data = {
        "voids": voids_data,
        "edges": edges_data,
        "metadata": metadata
    }

    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(visualization_data, f, indent=2)

    logger.info(f"Exported visualization data to {output_path}")
    logger.info(f"Exported {len(voids_data)} voids and {len(edges_data)} edges")

    return str(output_path)


def _extract_void_data(catalog_df: pd.DataFrame, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract individual void data for visualization.

    Parameters:
        catalog_df: Void catalog DataFrame
        results: Analysis results dictionary

    Returns:
        List of void dictionaries with position, properties, and clustering data
    """
    voids_data = []

    # Get clustering analysis results
    clustering_results = results.get("results", {}).get("clustering_analysis", {})
    network_analysis = results.get("results", {}).get("void_data", {}).get("network_analysis", {})

    # Extract local clustering coefficients
    local_ccs = network_analysis.get("local_clustering_coefficients", [])

    # Get global clustering coefficient for reference
    global_cc = network_analysis.get("clustering_coefficient", 0.0)

    # Process each void in the catalog
    for idx, void_row in catalog_df.iterrows():
        void_data = _extract_single_void_data(void_row, idx, local_ccs, global_cc)
        if void_data:
            voids_data.append(void_data)

    return voids_data


def _extract_single_void_data(void_row: pd.Series, idx: int,
                             local_ccs: List[float], global_cc: float) -> Optional[Dict[str, Any]]:
    """
    Extract data for a single void.

    Parameters:
        void_row: Single void row from catalog
        idx: Index in the catalog
        local_ccs: List of local clustering coefficients
        global_cc: Global clustering coefficient

    Returns:
        Void data dictionary or None if invalid
    """
    try:
        # Extract position (prefer Cartesian coordinates)
        # Try x_mpc, y_mpc, z_mpc first (from DESI)
        if pd.notna(void_row.get('x_mpc')) and pd.notna(void_row.get('y_mpc')) and pd.notna(void_row.get('z_mpc')):
            x, y, z = void_row['x_mpc'], void_row['y_mpc'], void_row['z_mpc']
        elif pd.notna(void_row.get('x')) and pd.notna(void_row.get('y')) and pd.notna(void_row.get('z')):
            x, y, z = void_row['x'], void_row['y'], void_row['z']
        else:
            # Try to convert from spherical if available
            if pd.notna(void_row.get('ra_deg')) and pd.notna(void_row.get('dec_deg')) and pd.notna(void_row.get('redshift')):
                # Convert spherical to Cartesian
                from astropy.cosmology import Planck18
                from astropy.coordinates import SkyCoord
                from astropy import units as u
                
                redshift = void_row['redshift']
                # Only convert if redshift is reasonable
                if 0 < redshift < 2:
                    coords = SkyCoord(
                        ra=void_row['ra_deg'] * u.deg,
                        dec=void_row['dec_deg'] * u.deg,
                        distance=Planck18.comoving_distance(redshift)
                    )
                    x, y, z = coords.cartesian.x.value, coords.cartesian.y.value, coords.cartesian.z.value
                else:
                    return None
            else:
                return None

        # Extract radius
        radius = void_row.get('radius_mpc', void_row.get('radius_eff', 10.0))
        if pd.isna(radius) or radius <= 0:
            radius = 10.0  # Default radius

        # Extract orientation (default to 0 if not available)
        orientation = void_row.get('orientation_deg', 0.0)
        if pd.isna(orientation):
            orientation = 0.0

        # Extract redshift (filter out corrupted values)
        redshift = void_row.get('redshift', 0.0)
        if pd.isna(redshift) or redshift > 10:
            # If redshift is missing or corrupted, try to compute from position
            distance = float((x**2 + y**2 + z**2)**0.5)
            if distance > 0:
                # Approximate redshift from comoving distance
                # For small z: d ≈ (c/H0) * z, so z ≈ d * H0 / c
                # H0 = 67.4 km/s/Mpc, c = 299792 km/s
                redshift = distance * 67.4 / 299792.0
            else:
                redshift = 0.0

        # Extract local clustering coefficient
        local_cc = global_cc  # Default to global if local not available
        if idx < len(local_ccs) and local_ccs[idx] is not None:
            local_cc = local_ccs[idx]

        # Determine survey source
        survey = _determine_survey_source(void_row)

        # Create void data dictionary
        void_data = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "radius": float(radius),
            "orientation": float(orientation),
            "redshift": float(redshift),
            "clustering": float(local_cc),
            "survey": survey
        }

        return void_data

    except Exception as e:
        logger.warning(f"Error extracting data for void {idx}: {e}")
        return None


def _determine_survey_source(void_row: pd.Series) -> str:
    """
    Determine which survey a void comes from.

    Parameters:
        void_row: Void data row

    Returns:
        Survey name string
    """
    # Check for survey column directly (added during processing)
    if 'survey' in void_row.index and pd.notna(void_row.get('survey')):
        return str(void_row['survey'])
    
    # Fallback: Check for survey-specific columns or naming patterns
    if 'clampitt' in str(void_row.get('source', '')).lower():
        return "SDSS_DR7_CLAMPITT"
    elif 'douglass' in str(void_row.get('source', '')).lower():
        return "SDSS_DR7_DOUGLASS"
    elif 'desi' in str(void_row.get('source', '')).lower():
        return "DESI_DR1"
    elif pd.notna(void_row.get('ra_deg')) and pd.notna(void_row.get('dec_deg')):
        # Try to infer from coordinate ranges
        ra = void_row.get('ra_deg', 0)
        dec = void_row.get('dec_deg', 0)
        # DESI has specific footprint patterns
        if -20 <= dec <= 90 and 0 <= ra <= 360:
            return "DESI_DR1"
        else:
            return "SDSS_DR7"
    else:
        return "UNKNOWN"


def _extract_network_edges(catalog_df: pd.DataFrame, results: Dict[str, Any]) -> List[List[int]]:
    """
    Extract network edges for visualization.

    Reconstructs the void network using the same linking length from the analysis,
    then exports all edges for client-side rendering.

    Parameters:
        catalog_df: Void catalog DataFrame
        results: Analysis results

    Returns:
        List of [source_idx, target_idx] edge pairs
    """
    logger.info("Generating network edges for visualization...")
    
    # Get linking length from results
    network_analysis = results.get("results", {}).get("void_data", {}).get("network_analysis", {})
    linking_length = network_analysis.get("linking_length", 60.0)
    n_edges_expected = network_analysis.get("n_edges", 0)
    
    logger.info(f"  Linking length: {linking_length:.1f} Mpc")
    logger.info(f"  Expected edges: {n_edges_expected:,}")
    
    # Get positions from catalog
    from pipeline.common.void_coordinates import get_cartesian_positions
    
    try:
        positions, filtered_catalog, _ = get_cartesian_positions(catalog_df)
        
        if positions is None or len(positions) == 0:
            logger.warning("  Could not extract positions for edge generation")
            return []
        
        logger.info(f"  Building network with {len(positions)} voids...")
        
        # Build distance matrix efficiently using chunking
        from scipy.spatial.distance import pdist, squareform
        
        # For large networks, compute distances in chunks to avoid memory issues
        n_voids = len(positions)
        edges = []
        
        if n_voids < 10000:
            # Small enough to compute all distances at once
            distances = squareform(pdist(positions))
            for i in range(n_voids):
                for j in range(i+1, n_voids):
                    if distances[i, j] <= linking_length:
                        edges.append([i, j])
        else:
            # Large network: use KDTree for efficient neighbor finding
            from scipy.spatial import cKDTree
            
            tree = cKDTree(positions)
            pairs = tree.query_pairs(r=linking_length, output_type='ndarray')
            
            edges = pairs.tolist()
        
        logger.info(f"  ✓ Generated {len(edges):,} edges")
        
        return edges
        
    except Exception as e:
        logger.error(f"  Failed to generate edges: {e}")
        return []


def _create_metadata(catalog_df: pd.DataFrame, results: Dict[str, Any],
                    voids_data: List[Dict], edges_data: List[List[int]]) -> Dict[str, Any]:
    """
    Create metadata for the visualization.

    Parameters:
        catalog_df: Original void catalog
        results: Analysis results
        voids_data: Exported voids data
        edges_data: Exported edges data

    Returns:
        Metadata dictionary
    """
    # Extract network analysis metadata
    network_analysis = results.get("results", {}).get("void_data", {}).get("network_analysis", {})

    metadata = {
        "n_voids": len(voids_data),
        "n_edges": len(edges_data),
        "linking_length": network_analysis.get("linking_length", 0),
        "global_clustering_coefficient": network_analysis.get("clustering_coefficient", 0),
        "mean_degree": network_analysis.get("mean_degree", 0),
        "survey_breakdown": results.get("results", {}).get("void_data", {}).get("survey_breakdown", {}),
        "eta_natural": 0.4430,  # Theoretical value
        "c_e8": 0.78125,  # E8×E8 substrate value
        "c_lcdm": 0.0,  # ΛCDM prediction
        "clustering_range": {
            "min": min(v["clustering"] for v in voids_data) if voids_data else 0,
            "max": max(v["clustering"] for v in voids_data) if voids_data else 0
        },
        "redshift_range": {
            "min": min(v["redshift"] for v in voids_data) if voids_data else 0,
            "max": max(v["redshift"] for v in voids_data) if voids_data else 0
        },
        "radius_range": {
            "min": min(v["radius"] for v in voids_data) if voids_data else 0,
            "max": max(v["radius"] for v in voids_data) if voids_data else 0
        }
    }

    return metadata
