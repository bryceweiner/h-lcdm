"""
Tests for H-ZOBOV Pipeline
==========================

Comprehensive test suite for H-ZOBOV void-finding algorithm.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from pipeline.voidfinder.zobov.zobov_pipeline import HZOBOVPipeline, HZOBOVPipelineError
from pipeline.voidfinder.zobov.zobov_parameters import HZOBOVParameters
from pipeline.voidfinder.zobov.zobov_core import ZOBOVCore, ZOBOVCoreError
from pipeline.voidfinder.zobov.hlcdm_integration import get_lambda_at_redshift, HZOBOVLambdaError
from pipeline.voidfinder.zobov.voronoi_tessellation import VoronoiTessellation
from pipeline.voidfinder.zobov.watershed import WatershedZoneFinder
from pipeline.voidfinder.zobov.zone_merger import ZoneMerger


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_galaxy_catalog():
    """Create sample galaxy catalog for testing."""
    np.random.seed(42)
    n_galaxies = 1000
    
    # Generate random positions in a cube
    positions = np.random.uniform(-50, 50, size=(n_galaxies, 3))
    
    # Add some redshift values
    redshifts = np.random.uniform(0.01, 0.15, size=n_galaxies)
    
    catalog = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'redshift': redshifts,
        'ra': np.random.uniform(0, 360, n_galaxies),
        'dec': np.random.uniform(-90, 90, n_galaxies),
    })
    
    return catalog


@pytest.fixture
def hzobov_parameters():
    """Create H-ZOBOV parameters for testing."""
    return HZOBOVParameters(
        output_name='test_output',
        z_min=0.0,
        z_max=0.2,
        use_hlcdm_lambda=True,
        batch_size=1000
    )


class TestHZOBOVParameters:
    """Test H-ZOBOV parameter configuration."""
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = HZOBOVParameters(
            output_name='test',
            z_min=0.0,
            z_max=1.0,
            batch_size=1000
        )
        params.validate()  # Should not raise
        
        # Invalid batch_size
        params.batch_size = -1
        with pytest.raises(ValueError):
            params.validate()
        
        # Invalid z_min
        params.batch_size = 1000
        params.z_min = -1
        with pytest.raises(ValueError):
            params.validate()
        
        # Invalid z_max
        params.z_min = 0.0
        params.z_max = 0.0
        with pytest.raises(ValueError):
            params.validate()
    
    def test_output_filename_generation(self):
        """Test output filename generation."""
        params = HZOBOVParameters(
            output_name='my_analysis',
            z_min=0.01,
            z_max=0.15,
            significance_ratio=0.2,
            min_void_volume=100.0
        )
        
        filename = params.get_output_filename_base(include_params=True)
        assert 'my_analysis' in filename
        assert 'zmin_0_01' in filename
        assert 'zmax_0_15' in filename
        assert 'sig_0_2' in filename
        assert 'minvol_100_0' in filename


class TestHLCDMIntegration:
    """Test H-ΛCDM Lambda integration."""
    
    def test_get_lambda_at_redshift_scalar(self):
        """Test Lambda calculation for scalar redshift."""
        z = 0.1
        lambda_z = get_lambda_at_redshift(z)
        
        assert isinstance(lambda_z, float)
        assert lambda_z > 0
        assert np.isfinite(lambda_z)
    
    def test_get_lambda_at_redshift_array(self):
        """Test Lambda calculation for array of redshifts."""
        z_array = np.array([0.0, 0.1, 0.2])
        lambda_array = get_lambda_at_redshift(z_array)
        
        assert isinstance(lambda_array, np.ndarray)
        assert len(lambda_array) == len(z_array)
        assert np.all(lambda_array > 0)
        assert np.all(np.isfinite(lambda_array))
    
    def test_get_lambda_invalid_redshift(self):
        """Test Lambda calculation with invalid redshift."""
        with pytest.raises(HZOBOVLambdaError):
            get_lambda_at_redshift(-1.0)


class TestVoronoiTessellation:
    """Test Voronoi tessellation stage."""
    
    def test_voronoi_tessellation_basic(self, sample_galaxy_catalog, hzobov_parameters):
        """Test basic Voronoi tessellation."""
        voronoi = VoronoiTessellation(hzobov_parameters, use_mps=False)  # Use CPU for testing
        
        result = voronoi.process(sample_galaxy_catalog)
        
        assert 'voronoi' in result
        assert 'volumes' in result
        assert 'densities' in result
        assert 'positions' in result
        assert len(result['volumes']) == len(sample_galaxy_catalog)
        assert len(result['densities']) == len(sample_galaxy_catalog)
        assert np.all(result['volumes'] > 0)
        assert np.all(result['densities'] > 0)
    
    def test_voronoi_insufficient_points(self, hzobov_parameters):
        """Test Voronoi tessellation with insufficient points."""
        catalog = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [0, 1, 2]
        })
        
        voronoi = VoronoiTessellation(hzobov_parameters, use_mps=False)
        
        with pytest.raises(Exception):  # Should raise error for < 4 points
            voronoi.process(catalog)


class TestWatershedZoneFinder:
    """Test watershed zone finding."""
    
    def test_watershed_basic(self, sample_galaxy_catalog, hzobov_parameters):
        """Test basic watershed zone finding."""
        # First compute Voronoi
        voronoi = VoronoiTessellation(hzobov_parameters, use_mps=False)
        voronoi_data = voronoi.process(sample_galaxy_catalog)
        
        # Then watershed
        watershed = WatershedZoneFinder(use_mps=False)
        result = watershed.process(
            voronoi_data['densities'],
            voronoi_data['positions']
        )
        
        assert 'zone_ids' in result
        assert 'n_zones' in result
        assert len(result['zone_ids']) == len(sample_galaxy_catalog)
        assert result['n_zones'] > 0
        assert np.all(result['zone_ids'] >= 0)


class TestZoneMerger:
    """Test zone merging."""
    
    def test_zone_merger_basic(self, sample_galaxy_catalog, hzobov_parameters):
        """Test basic zone merging."""
        # Run Voronoi and watershed first
        voronoi = VoronoiTessellation(hzobov_parameters, use_mps=False)
        voronoi_data = voronoi.process(sample_galaxy_catalog)
        
        watershed = WatershedZoneFinder(use_mps=False)
        watershed_data = watershed.process(
            voronoi_data['densities'],
            voronoi_data['positions']
        )
        
        # Then merge
        merger = ZoneMerger(hzobov_parameters)
        result = merger.process(
            watershed_data,
            voronoi_data['densities'],
            voronoi_data['volumes'],
            voronoi_data['positions'],
            sample_galaxy_catalog['redshift'].values
        )
        
        assert 'void_ids' in result
        assert 'n_voids' in result
        assert len(result['void_ids']) == len(sample_galaxy_catalog)
        assert result['n_voids'] > 0


class TestZOBOVCore:
    """Test ZOBOV core algorithm."""
    
    def test_zobov_core_complete(self, sample_galaxy_catalog, hzobov_parameters):
        """Test complete ZOBOV algorithm."""
        core = ZOBOVCore(hzobov_parameters)
        
        result = core.process(sample_galaxy_catalog)
        
        assert 'void_catalog' in result
        assert 'n_voids' in result
        assert isinstance(result['void_catalog'], pd.DataFrame)
        assert len(result['void_catalog']) == result['n_voids']
        
        # Check catalog columns
        required_cols = ['void_id', 'x', 'y', 'z', 'radius_mpc', 'volume_mpc3']
        for col in required_cols:
            assert col in result['void_catalog'].columns


class TestHZOBOVPipeline:
    """Test H-ZOBOV pipeline."""
    
    def test_pipeline_initialization(self, temp_output_dir):
        """Test pipeline initialization."""
        pipeline = HZOBOVPipeline(str(temp_output_dir))
        
        assert pipeline.name == 'hzobov'
        assert pipeline.base_output_dir == temp_output_dir
    
    def test_pipeline_missing_output_name(self, temp_output_dir, sample_galaxy_catalog):
        """Test pipeline fails without output_name."""
        pipeline = HZOBOVPipeline(str(temp_output_dir))
        
        context = {
            'catalog': 'sdss_dr16',
            'z_min': 0.0,
            'z_max': 0.2
        }
        
        with pytest.raises(HZOBOVPipelineError):
            pipeline.run(context)
    
    def test_pipeline_with_output_name(self, temp_output_dir):
        """Test pipeline accepts output_name."""
        pipeline = HZOBOVPipeline(str(temp_output_dir))
        
        context = {
            'output_name': 'test_run',
            'catalog': 'sdss_dr16',
            'z_min': 0.0,
            'z_max': 0.2
        }
        
        # This will fail at catalog loading, but should pass output_name check
        # We can't easily test full pipeline without actual catalog data
        try:
            pipeline.run(context)
        except HZOBOVPipelineError as e:
            # Expected to fail at catalog loading, but should not fail at output_name check
            assert 'output_name' not in str(e).lower() or 'mandatory' not in str(e).lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

