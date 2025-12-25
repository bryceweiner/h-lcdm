"""
Test suite for void visualization components.

Tests data export, HTML generation, and statistical figures.
"""

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import visualization modules
from pipeline.void.visualization.data_export import export_void_visualization_data
from pipeline.void.visualization.void_map import generate_void_3d_map_html
from pipeline.void.visualization.statistical_figures import generate_void_statistical_figures


class TestVoidVisualizationDataExport:
    """Test void data export functionality."""

    def test_export_with_mock_data(self):
        """Test data export with mock void catalog and results."""
        # Create mock void catalog
        mock_catalog = pd.DataFrame({
            'x': [100.0, 200.0, 150.0],
            'y': [-50.0, 75.0, 25.0],
            'z': [200.0, 150.0, 175.0],
            'radius_mpc': [15.0, 20.0, 12.0],
            'orientation_deg': [45.0, 90.0, 30.0],
            'redshift': [0.5, 0.7, 0.6],
            'source': ['clampitt', 'clampitt', 'clampitt']
        })

        # Create mock results
        mock_results = {
            "results": {
                "void_data": {
                    "network_analysis": {
                        "local_clustering_coefficients": [0.5, 0.7, 0.3],
                        "clustering_coefficient": 0.51,
                        "linking_length": 60.0,
                        "mean_degree": 22.0,
                        "n_nodes": 3,
                        "n_edges": 33
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save mock data to temporary files
            catalog_path = Path(temp_dir) / "mock_catalog.pkl"
            results_path = Path(temp_dir) / "mock_results.json"
            output_path = Path(temp_dir) / "test_output.json"

            mock_catalog.to_pickle(catalog_path)
            with open(results_path, 'w') as f:
                json.dump(mock_results, f)

            # Test export
            result_path = export_void_visualization_data(
                void_catalog_path=str(catalog_path),
                results_path=str(results_path),
                output_path=str(output_path)
            )

            # Verify output file exists
            assert Path(result_path).exists()

            # Load and verify content
            with open(result_path, 'r') as f:
                data = json.load(f)

            assert 'voids' in data
            assert 'edges' in data
            assert 'metadata' in data

            assert len(data['voids']) == 3
            assert isinstance(data['edges'], list)
            assert data['metadata']['n_voids'] == 3

            # Verify void data structure
            void1 = data['voids'][0]
            assert 'x' in void1
            assert 'y' in void1
            assert 'z' in void1
            assert 'radius' in void1
            assert 'orientation' in void1
            assert 'clustering' in void1
            assert 'survey' in void1

    def test_export_handles_missing_data(self):
        """Test export handles missing or invalid data gracefully."""
        # Create catalog with some missing data
        mock_catalog = pd.DataFrame({
            'x': [100.0, np.nan, 150.0],
            'y': [-50.0, 75.0, np.nan],
            'z': [200.0, 150.0, 175.0],
            'radius_mpc': [15.0, 20.0, 12.0],
            'orientation_deg': [45.0, np.nan, 30.0],
            'redshift': [0.5, 0.7, 0.6]
        })

        mock_results = {
            "results": {
                "void_data": {
                    "network_analysis": {
                        "local_clustering_coefficients": [0.5, 0.7, 0.3],
                        "clustering_coefficient": 0.51
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "mock_catalog.pkl"
            results_path = Path(temp_dir) / "mock_results.json"
            output_path = Path(temp_dir) / "test_output.json"

            mock_catalog.to_pickle(catalog_path)
            with open(results_path, 'w') as f:
                json.dump(mock_results, f)

            # Should not raise exception
            result_path = export_void_visualization_data(
                void_catalog_path=str(catalog_path),
                results_path=str(results_path),
                output_path=str(output_path)
            )

            # Load and verify content
            with open(result_path, 'r') as f:
                data = json.load(f)

            # Should have fewer voids due to invalid data
            assert len(data['voids']) <= 3


class TestVoidVisualizationHTML:
    """Test HTML/Three.js map generation."""

    def test_html_generation_with_mock_data(self):
        """Test HTML generation with mock visualization data."""
        mock_viz_data = {
            "voids": [
                {"x": 100.0, "y": -50.0, "z": 200.0, "radius": 15.0,
                 "orientation": 45.0, "clustering": 0.5, "survey": "SDSS_DR7"},
                {"x": 200.0, "y": 75.0, "z": 150.0, "radius": 20.0,
                 "orientation": 90.0, "clustering": 0.7, "survey": "DESI_DR1"}
            ],
            "edges": [[0, 1]],
            "metadata": {
                "n_voids": 2,
                "n_edges": 1,
                "global_clustering_coefficient": 0.51,
                "linking_length": 60.0,
                "eta_natural": 0.443,
                "c_e8": 0.78125,
                "c_lcdm": 0.0
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "mock_viz_data.json"
            output_path = Path(temp_dir) / "test_map.html"

            with open(data_path, 'w') as f:
                json.dump(mock_viz_data, f)

            # Test HTML generation
            result_path = generate_void_3d_map_html(
                data_path=str(data_path),
                output_path=str(output_path)
            )

            # Verify output file exists
            assert Path(result_path).exists()

            # Verify HTML content
            with open(result_path, 'r') as f:
                html_content = f.read()

            assert '<!DOCTYPE html>' in html_content
            assert 'Three.js' in html_content
            assert 'voidsData' in html_content
            assert 'edgesData' in html_content
            assert 'Cosmic Void Network' in html_content

    def test_html_handles_missing_edges(self):
        """Test HTML generation when no edges are present."""
        mock_viz_data = {
            "voids": [
                {"x": 100.0, "y": -50.0, "z": 200.0, "radius": 15.0,
                 "orientation": 45.0, "clustering": 0.5, "survey": "SDSS_DR7"}
            ],
            "edges": [],
            "metadata": {
                "n_voids": 1,
                "n_edges": 0,
                "global_clustering_coefficient": 0.5
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "mock_viz_data.json"
            output_path = Path(temp_dir) / "test_map.html"

            with open(data_path, 'w') as f:
                json.dump(mock_viz_data, f)

            # Should not raise exception
            result_path = generate_void_3d_map_html(
                data_path=str(data_path),
                output_path=str(output_path)
            )

            assert Path(result_path).exists()


class TestVoidStatisticalFigures:
    """Test statistical figures generation."""

    def test_figure_generation_with_mock_data(self):
        """Test statistical figure generation with mock data."""
        # Create mock catalog
        mock_catalog = pd.DataFrame({
            'x': [100.0, 200.0, 150.0, 300.0, 250.0],
            'y': [-50.0, 75.0, 25.0, 100.0, 50.0],
            'z': [200.0, 150.0, 175.0, 125.0, 180.0],
            'radius_mpc': [15.0, 20.0, 12.0, 18.0, 14.0],
            'orientation_deg': [45.0, 90.0, 30.0, 60.0, 120.0],
            'redshift': [0.5, 0.7, 0.6, 0.8, 0.4],
            'source': ['clampitt', 'clampitt', 'douglass', 'clampitt', 'clampitt'],
            'ra_deg': [120.0, 130.0, 140.0, 150.0, 160.0],
            'dec_deg': [-10.0, 5.0, 15.0, 25.0, 35.0]
        })

        # Create mock results
        mock_results = {
            "results": {
                "clustering_analysis": {
                    "observed_clustering_coefficient": 0.5102047579982437,
                    "observed_clustering_std": 0.10973130384186014,
                    "clustering_comparison": {
                        "thermodynamic_efficiency": {"sigma": 2.1},
                        "lcmd": {"sigma": -5.2}
                    },
                    "model_comparison": {
                        "overall_scores": {
                            "hlcdm_combined": 0.8,
                            "lcmd_connectivity_only": 1.2
                        },
                        "best_model": "H-ΛCDM"
                    }
                },
                "void_data": {
                    "network_analysis": {
                        "local_clustering_coefficients": [0.5, 0.7, 0.3, 0.6, 0.4],
                        "clustering_coefficient": 0.51,
                        "linking_length": 60.0,
                        "mean_degree": 22.0,
                        "n_nodes": 5,
                        "n_edges": 55
                    },
                    "total_voids": 5,
                    "survey_breakdown": {
                        "SDSS_DR7_CLAMPITT": 4,
                        "SDSS_DR7_DOUGLASS": 1
                    }
                },
                "validation": {
                    "bootstrap": {
                        "bootstrap_mean": 0.508,
                        "bootstrap_std": 0.012,
                        "z_score": 0.15,
                        "passed": True
                    },
                    "null_hypothesis": {
                        "null_mean": 0.002,
                        "null_std": 0.001,
                        "p_value": 1e-15,
                        "z_score": 340.0,
                        "passed": True
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "mock_catalog.pkl"
            results_path = Path(temp_dir) / "mock_results.json"
            figures_dir = Path(temp_dir) / "figures"

            mock_catalog.to_pickle(catalog_path)
            with open(results_path, 'w') as f:
                json.dump(mock_results, f)

            # Test figure generation
            figure_paths = generate_void_statistical_figures(
                void_catalog_path=str(catalog_path),
                results_path=str(results_path),
                output_dir=str(figures_dir)
            )

            # Verify output files exist
            assert isinstance(figure_paths, dict)
            assert len(figure_paths) > 0

            # Check that key figures were generated
            expected_figures = [
                'clustering_distribution',
                'redshift_distribution',
                'size_distribution',
                'network_degree',
                'bootstrap_validation',
                'null_hypothesis',
                'model_comparison',
                'spatial_distribution'
            ]

            for fig_name in expected_figures:
                assert fig_name in figure_paths
                assert Path(figure_paths[fig_name]).exists()

    def test_figure_handles_missing_data(self):
        """Test figure generation handles missing data gracefully."""
        # Create minimal mock data
        mock_catalog = pd.DataFrame({
            'radius_mpc': [15.0, 20.0],
            'redshift': [0.5, 0.7]
        })

        mock_results = {
            "results": {
                "clustering_analysis": {
                    "observed_clustering_coefficient": 0.5
                },
                "void_data": {
                    "network_analysis": {
                        "local_clustering_coefficients": [0.5, 0.7]
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "mock_catalog.pkl"
            results_path = Path(temp_dir) / "mock_results.json"
            figures_dir = Path(temp_dir) / "figures"

            mock_catalog.to_pickle(catalog_path)
            with open(results_path, 'w') as f:
                json.dump(mock_results, f)

            # Should not raise exception
            figure_paths = generate_void_statistical_figures(
                void_catalog_path=str(catalog_path),
                results_path=str(results_path),
                output_dir=str(figures_dir)
            )

            assert isinstance(figure_paths, dict)


if __name__ == '__main__':
    pytest.main([__file__])
