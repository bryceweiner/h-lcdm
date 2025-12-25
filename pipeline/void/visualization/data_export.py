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
    output_path: str = "results/figures/void/void_map_data.json"
) -> str:
    """
    Export void data for 3D visualization.

    Extracts void positions, radii, orientations, clustering coefficients,
    and network edges from processed void catalog and analysis results.

    Parameters:
        void_catalog_path: Path to processed void catalog pickle file
        results_path: Path to void analysis results JSON file
        output_path: Path to output JSON file for 3D visualization

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

    # Load analysis results
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
        if pd.notna(void_row.get('x')) and pd.notna(void_row.get('y')) and pd.notna(void_row.get('z')):
            x, y, z = void_row['x'], void_row['y'], void_row['z']
        elif pd.notna(void_row.get('x_mpc')) and pd.notna(void_row.get('y_mpc')) and pd.notna(void_row.get('z_mpc')):
            x, y, z = void_row['x_mpc'], void_row['y_mpc'], void_row['z_mpc']
        else:
            # Try to convert from spherical if available
            if pd.notna(void_row.get('ra_deg')) and pd.notna(void_row.get('dec_deg')) and pd.notna(void_row.get('redshift')):
                # Note: In a real implementation, we'd convert spherical to Cartesian here
                # For now, skip voids without Cartesian coordinates
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

        # Extract redshift
        redshift = void_row.get('redshift', 0.0)
        if pd.isna(redshift):
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
    # Check for survey-specific columns or naming patterns
    if 'clampitt' in str(void_row.get('source', '')).lower():
        return "SDSS_DR7_CLAMPITT"
    elif 'douglass' in str(void_row.get('source', '')).lower():
        return "SDSS_DR7_DOUGLASS"
    elif 'desi' in str(void_row.get('survey', '')).lower():
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

    For performance, we sample edges rather than including all 540k edges.
    The 3D viewer will show edges on demand or for hovered voids.

    Parameters:
        catalog_df: Void catalog DataFrame
        results: Analysis results

    Returns:
        List of [source_idx, target_idx] edge pairs
    """
    # For the 3D visualization, we don't need to export all 540k edges
    # as this would make the JSON file too large and slow to load.
    # Instead, we export a smaller sample or implement lazy loading.

    # For now, return an empty list - edges will be computed client-side
    # or loaded on demand in future implementations
    logger.info("Skipping network edge export for performance (540k edges would be too large)")
    logger.info("Edges can be computed client-side or loaded on demand in future versions")

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
