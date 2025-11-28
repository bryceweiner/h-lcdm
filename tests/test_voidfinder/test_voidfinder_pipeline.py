"""
Tests for VoidFinder Pipeline
==============================
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from pipeline.voidfinder import VoidFinderPipeline
from pipeline.voidfinder.checkpoint import CheckpointManager
from pipeline.voidfinder.galaxy_catalogs.catalog_registry import CatalogRegistry


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def voidfinder_pipeline(temp_dir):
    """Create VoidFinderPipeline instance for testing."""
    return VoidFinderPipeline(output_dir=str(temp_dir / "results"))


def test_checkpoint_manager(temp_dir):
    """Test checkpoint manager functionality."""
    checkpoint_dir = temp_dir / "checkpoints"
    manager = CheckpointManager(checkpoint_dir, "test")
    
    # Test save and load
    data = {'batch_id': 5, 'total_batches': 10, 'rows_downloaded': 50000}
    path = manager.save_checkpoint('download', data)
    
    assert path.exists()
    
    loaded = manager.load_checkpoint('download')
    assert loaded is not None
    assert loaded['batch_id'] == 5
    assert loaded['total_batches'] == 10
    
    # Test clear
    manager.clear_checkpoint('download')
    assert not manager.has_checkpoint('download')


def test_catalog_registry():
    """Test catalog registry."""
    available = CatalogRegistry.list_available()
    assert 'sdss_dr16' in available
    assert 'sdss_dr7' in available
    
    # Test get
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = CatalogRegistry.get('sdss_dr16', Path(tmpdir), Path(tmpdir))
        assert catalog.catalog_name == 'sdss_dr16'
        
        # Test DR7 as well
        catalog_dr7 = CatalogRegistry.get('sdss_dr7', Path(tmpdir), Path(tmpdir))
        assert catalog_dr7.catalog_name == 'sdss_dr7'


def test_voidfinder_pipeline_init(voidfinder_pipeline):
    """Test pipeline initialization."""
    assert voidfinder_pipeline.name == "voidfinder"
    assert 'sdss_dr16' in voidfinder_pipeline.available_catalogs
    assert 'sdss_dr7' in voidfinder_pipeline.available_catalogs


def test_voidfinder_pipeline_validate(voidfinder_pipeline, temp_dir):
    """Test pipeline validation."""
    # Ensure catalog file doesn't exist initially
    void_file = voidfinder_pipeline.processed_data_dir / "voidfinder_sdss_dr16.pkl"
    if void_file.exists():
        void_file.unlink()
    
    # Test with non-existent catalog
    result = voidfinder_pipeline.validate({'catalog': 'sdss_dr16'})
    assert 'valid' in result
    assert result['valid'] == False  # Catalog doesn't exist yet
    
    # Create a mock void catalog
    void_file.parent.mkdir(parents=True, exist_ok=True)
    
    mock_voids = pd.DataFrame({
        'x': np.random.uniform(-1000, 1000, 100),
        'y': np.random.uniform(-1000, 1000, 100),
        'z': np.random.uniform(-1000, 1000, 100),
        'radius_mpc': np.random.uniform(5, 50, 100),
        'redshift': np.random.uniform(0.01, 0.2, 100)
    })
    mock_voids.to_pickle(void_file)
    
    # Test validation with valid catalog
    result = voidfinder_pipeline.validate({'catalog': 'sdss_dr16'})
    assert result['valid'] == True
    assert result['n_voids'] == 100


def test_convert_to_comoving(voidfinder_pipeline):
    """Test coordinate conversion."""
    # Create mock galaxy catalog
    galaxies = pd.DataFrame({
        'ra': np.random.uniform(0, 360, 100),
        'dec': np.random.uniform(-90, 90, 100),
        'z': np.random.uniform(0.01, 0.2, 100)
    })
    
    # Convert to comoving
    coords = voidfinder_pipeline._convert_to_comoving(galaxies)
    
    assert 'x' in coords.columns
    assert 'y' in coords.columns
    assert 'z' in coords.columns
    assert len(coords) == 100


def test_remove_duplicate_voids(voidfinder_pipeline):
    """Test duplicate void removal."""
    # Create mock voids with duplicates
    voids = pd.DataFrame({
        'x': [0, 1, 100, 101, 200],
        'y': [0, 1, 0, 1, 0],
        'z': [0, 1, 0, 1, 0],
        'radius_mpc': [10, 10, 10, 10, 10]
    })
    
    # First two are duplicates (distance < 5 Mpc)
    deduplicated = voidfinder_pipeline._remove_duplicate_voids(voids, min_separation=5.0)
    
    # Should have fewer voids
    assert len(deduplicated) <= len(voids)
    assert len(deduplicated) >= 3  # At least 3 unique voids


def test_generate_summary(voidfinder_pipeline):
    """Test summary generation."""
    voids = pd.DataFrame({
        'radius_mpc': np.random.uniform(5, 50, 100),
        'redshift': np.random.uniform(0.01, 0.2, 100)
    })
    
    summary = voidfinder_pipeline._generate_summary(voids)
    
    assert 'n_voids' in summary
    assert summary['n_voids'] == 100
    assert 'radius_stats' in summary
    assert 'redshift_stats' in summary

