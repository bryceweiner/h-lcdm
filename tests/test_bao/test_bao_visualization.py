"""
Tests for BAO visualization functionality.

Tests the high-quality BAO scale visualization scripts.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from scripts.bao_visualization import BAOVisualization


class TestBAOVisualization:
    """Test BAO visualization functionality."""

    @pytest.fixture
    def viz(self):
        """Create visualization instance."""
        return BAOVisualization()

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return tmp_path

    def test_visualization_initialization(self, viz):
        """Test visualization class initialization."""
        assert hasattr(viz, 'data_loader')
        assert hasattr(viz, 'bao_pipeline')
        assert hasattr(viz, 'survey_colors')
        assert hasattr(viz, 'survey_names')

    def test_data_loading(self, viz):
        """Test BAO data loading for visualizations."""
        bao_data = viz.load_all_bao_data()

        assert isinstance(bao_data, dict)
        assert len(bao_data) > 0  # Should load at least some surveys

        # Check that loaded data has expected structure
        for survey_name, survey_data in bao_data.items():
            assert 'measurements' in survey_data
            assert 'name' in survey_data
            assert isinstance(survey_data['measurements'], list)

            if survey_data['measurements']:
                measurement = survey_data['measurements'][0]
                assert 'z' in measurement
                assert 'value' in measurement
                assert 'error' in measurement

    def test_theoretical_predictions(self, viz):
        """Test theoretical prediction calculations."""
        z_test = np.array([0.5, 1.0, 1.5])
        hlcdm_predictions, lcdm_predictions = viz.calculate_theoretical_predictions(z_test)

        # Check both prediction arrays
        assert isinstance(hlcdm_predictions, np.ndarray)
        assert isinstance(lcdm_predictions, np.ndarray)
        assert len(hlcdm_predictions) == len(z_test)
        assert len(lcdm_predictions) == len(z_test)

        # Should be positive values
        valid_hlcdm = hlcdm_predictions[~np.isnan(hlcdm_predictions)]
        valid_lcdm = lcdm_predictions[~np.isnan(lcdm_predictions)]
        assert all(p > 0 for p in valid_hlcdm)
        assert all(p > 0 for p in valid_lcdm)

        # Should increase with redshift (distance increases)
        assert hlcdm_predictions[1] > hlcdm_predictions[0]  # z=1.0 > z=0.5
        assert hlcdm_predictions[2] > hlcdm_predictions[1]  # z=1.5 > z=1.0
        assert lcdm_predictions[1] > lcdm_predictions[0]
        assert lcdm_predictions[2] > lcdm_predictions[1]

        # ΛCDM should be higher than H-ΛCDM (smaller r_s in denominator)
        assert all(lcdm_predictions > hlcdm_predictions)
        
        # Difference should be consistent (~2.17%)
        ratio = lcdm_predictions[0] / hlcdm_predictions[0]
        assert 1.02 < ratio < 1.03  # Should be around 150.71/147.5 ≈ 1.0218

    def test_plot_creation(self, viz, temp_output_dir):
        """Test plot creation without saving."""
        # This should not raise exceptions
        fig, ax = viz.create_bao_scale_plot(save_path=None)

        assert fig is not None
        assert ax is not None

        # Check that axes have expected elements
        assert ax.get_xlabel() == 'Redshift (z)'
        assert 'D$_M$(z)/r$_d$' in ax.get_ylabel()
        assert 'BAO Scale' in ax.get_title()

        plt.close(fig)

    def test_plot_saving(self, viz, temp_output_dir):
        """Test plot saving functionality."""
        output_path = temp_output_dir / 'test_bao_plot.png'

        fig, ax = viz.create_bao_scale_plot(save_path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File has content

        plt.close(fig)

    def test_comparison_plot_creation(self, viz, temp_output_dir):
        """Test comparison plot creation."""
        fig, (ax1, ax2) = viz.create_comparison_plot(save_path=None)

        assert fig is not None
        assert len((ax1, ax2)) == 2

        # Check main plot
        assert ax1.get_xlabel() == 'Redshift (z)'
        assert 'BAO Scale' in ax1.get_title()

        # Check residual plot
        assert 'Residual' in ax2.get_title()
        assert ax2.get_xlabel() == 'Redshift (z)'

        plt.close(fig)

    def test_comparison_plot_saving(self, viz, temp_output_dir):
        """Test comparison plot saving."""
        output_path = temp_output_dir / 'test_comparison_plot.png'

        fig, (ax1, ax2) = viz.create_comparison_plot(save_path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        plt.close(fig)

    def test_survey_color_mapping(self, viz):
        """Test survey color mapping."""
        # All expected surveys should have colors
        expected_surveys = ['boss_dr12', 'desi', 'eboss', 'wigglez', 'sixdfgs',
                           'sdss_mgs', 'sdss_dr7', '2dfgrs', 'des_y1', 'des_y3']

        for survey in expected_surveys:
            assert survey in viz.survey_colors
            assert survey in viz.survey_names

            # Colors should be valid hex colors
            color = viz.survey_colors[survey]
            assert color.startswith('#')
            assert len(color) == 7

    def test_main_script_execution(self, temp_output_dir):
        """Test main script execution."""
        import subprocess
        import sys

        # Change to project directory
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'bao_visualization.py'

        # Run the script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], cwd=temp_output_dir, capture_output=True, text=True)

        # Should complete successfully
        assert result.returncode == 0

        # Should create output files
        figures_dir = temp_output_dir / 'results' / 'figures'
        if figures_dir.exists():
            plot_files = list(figures_dir.glob('*.png'))
            assert len(plot_files) >= 2  # Should create at least 2 plots

    def test_visualization_data_integrity(self, viz):
        """Test that visualization data maintains physical integrity."""
        bao_data = viz.load_all_bao_data()

        for survey_name, survey_data in bao_data.items():
            measurements = survey_data.get('measurements', [])

            for measurement in measurements:
                # Redshifts should be reasonable (0 < z < 5)
                assert 0 < measurement['z'] < 5

                # Values should be positive and reasonable (D_M/r_d < 50 Mpc)
                assert 0 < measurement['value'] < 50

                # Errors should be positive and reasonable
                assert 0 < measurement.get('error', 1) < 10

                # Error should be smaller than value
                assert measurement.get('error', 0) < measurement['value']
