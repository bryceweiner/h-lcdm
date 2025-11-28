"""
Tests for SDSS DR7 Catalog Provider
====================================
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from pipeline.voidfinder.galaxy_catalogs.catalog_registry import CatalogRegistry
from pipeline.voidfinder.galaxy_catalogs.sdss_dr7_catalog import SDSSDR7Catalog


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


def test_catalog_registry_includes_dr7():
    """Test that catalog registry includes sdss_dr7."""
    available = CatalogRegistry.list_available()
    assert 'sdss_dr7' in available


def test_catalog_registry_get_dr7(temp_dir):
    """Test getting SDSS DR7 catalog from registry."""
    catalog = CatalogRegistry.get('sdss_dr7', temp_dir, temp_dir)
    assert isinstance(catalog, SDSSDR7Catalog)
    assert catalog.catalog_name == 'sdss_dr7'


def test_sdss_dr7_catalog_init(temp_dir):
    """Test SDSS DR7 catalog initialization."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    assert catalog.catalog_name == 'sdss_dr7'
    assert catalog.batch_size == 50000
    assert catalog.cache_file == temp_dir / "sdss_dr7" / "galaxies_full.pkl"
    assert catalog.cache_file.parent.exists()


def test_sdss_dr7_catalog_required_columns(temp_dir):
    """Test required columns property."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    required = catalog.required_columns
    
    assert isinstance(required, list)
    assert 'ra' in required
    assert 'dec' in required
    assert 'z' in required
    assert 'objid' in required


def test_sdss_dr7_catalog_load_from_cache(temp_dir):
    """Test loading catalog from cache."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Create mock cached catalog
    mock_data = pd.DataFrame({
        'ra': np.random.uniform(0, 360, 100),
        'dec': np.random.uniform(-90, 90, 100),
        'z': np.random.uniform(0.01, 0.2, 100),
        'objid': np.arange(100),
        'r_mag': np.random.uniform(15, 22, 100)
    })
    
    catalog.cache_file.parent.mkdir(parents=True, exist_ok=True)
    mock_data.to_pickle(catalog.cache_file)
    
    # Load from cache
    loaded = catalog.load(use_cache=True)
    
    assert loaded is not None
    assert len(loaded) == 100
    assert 'ra' in loaded.columns
    assert 'dec' in loaded.columns
    assert 'z' in loaded.columns
    assert 'objid' in loaded.columns


def test_sdss_dr7_catalog_load_no_cache(temp_dir):
    """Test loading when cache doesn't exist."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Ensure cache file doesn't exist
    if catalog.cache_file.exists():
        catalog.cache_file.unlink()
    
    loaded = catalog.load(use_cache=True)
    assert loaded is None


def test_sdss_dr7_catalog_load_cache_disabled(temp_dir):
    """Test loading with cache disabled."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Create mock cached catalog
    mock_data = pd.DataFrame({
        'ra': np.random.uniform(0, 360, 100),
        'dec': np.random.uniform(-90, 90, 100),
        'z': np.random.uniform(0.01, 0.2, 100),
        'objid': np.arange(100)
    })
    
    catalog.cache_file.parent.mkdir(parents=True, exist_ok=True)
    mock_data.to_pickle(catalog.cache_file)
    
    # Load with cache disabled
    loaded = catalog.load(use_cache=False)
    assert loaded is None


def test_sdss_dr7_catalog_validate_columns(temp_dir):
    """Test column validation."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Valid DataFrame
    valid_df = pd.DataFrame({
        'ra': [100.0],
        'dec': [45.0],
        'z': [0.1],
        'objid': [12345]
    })
    assert catalog.validate_columns(valid_df) == True
    
    # Invalid DataFrame (missing columns)
    invalid_df = pd.DataFrame({
        'ra': [100.0],
        'dec': [45.0]
        # Missing z and objid
    })
    with pytest.raises(ValueError):
        catalog.validate_columns(invalid_df)


def test_sdss_dr7_catalog_download_uses_data_release_7(temp_dir):
    """Test that SDSS DR7 catalog uses data_release=7 parameter."""
    import inspect
    from pipeline.voidfinder.galaxy_catalogs.sdss_dr7_catalog import SDSSDR7Catalog
    
    # Verify data_release=7 is in the source code
    source = inspect.getsource(SDSSDR7Catalog.download)
    assert 'data_release=7' in source or 'data_release= 7' in source, \
        "SDSS DR7 catalog should use data_release=7 in query_sql call"
    
    # Also verify it's not using the default (which would be DR16/DR17)
    assert 'data_release=7' in source, "Must explicitly set data_release=7 for DR7"


@patch('pipeline.voidfinder.galaxy_catalogs.sdss_dr7_catalog.SDSS')
def test_sdss_dr7_catalog_download_empty_result(mock_sdss, temp_dir):
    """Test download when query returns empty result."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Mock empty SDSS query result
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = pd.DataFrame()
    mock_sdss.query_sql.return_value = mock_result
    
    # Mock checkpoint manager
    mock_checkpoint = Mock()
    mock_checkpoint.load_checkpoint.return_value = None
    
    # Download should handle empty result gracefully
    # In actual implementation, this would mark as completed
    # For test, we'll verify it doesn't crash
    with pytest.raises(RuntimeError, match="No data downloaded"):
        catalog.download(
            checkpoint_manager=mock_checkpoint,
            z_min=0.01,
            z_max=0.2,
            mag_limit=22.0,
            force_redownload=False
        )


def test_sdss_dr7_catalog_download_no_astroquery(temp_dir):
    """Test download when astroquery is not available."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Temporarily set ASTROQUERY_AVAILABLE to False
    original_value = catalog.__class__.__module__
    with patch('pipeline.voidfinder.galaxy_catalogs.sdss_dr7_catalog.ASTROQUERY_AVAILABLE', False):
        with pytest.raises(ImportError, match="astroquery not available"):
            catalog.download(
                checkpoint_manager=None,
                z_min=0.01,
                z_max=0.2,
                mag_limit=22.0,
                force_redownload=False
            )


def test_sdss_dr7_catalog_get_volume_limited_sample(temp_dir):
    """Test volume-limited sample filtering."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'ra': np.random.uniform(0, 360, 1000),
        'dec': np.random.uniform(-90, 90, 1000),
        'z': np.random.uniform(0.01, 0.3, 1000),
        'objid': np.arange(1000),
        'magnitude': np.random.uniform(15, 22, 1000)
    })
    
    # Filter to volume-limited sample
    filtered = catalog.get_volume_limited_sample(test_df, z_max=0.2, mag_limit=21.0)
    
    assert len(filtered) <= len(test_df)
    assert all(filtered['z'] <= 0.2)
    assert all(filtered['magnitude'] <= 21.0)


def test_sdss_dr7_catalog_cache_file_path(temp_dir):
    """Test that cache file path is correct."""
    catalog = SDSSDR7Catalog(temp_dir, temp_dir)
    
    expected_path = temp_dir / "sdss_dr7" / "galaxies_full.pkl"
    assert catalog.cache_file == expected_path

