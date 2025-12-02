"""
Unit tests for CMB data loading (COBE and WMAP)
===============================================

Tests for loading COBE DMR and WMAP power spectrum data.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import requests
from io import BytesIO

from data.loader import DataLoader, DataUnavailableError


class TestCOBELoader:
    """Test COBE DMR data loading."""

    @pytest.fixture
    def data_loader(self, tmp_path):
        """Create a DataLoader instance with temporary directories."""
        return DataLoader(
            downloaded_data_dir=str(tmp_path / "downloaded_data"),
            processed_data_dir=str(tmp_path / "processed_data"),
            use_cache=False
        )

    def test_load_cobe_success(self, data_loader):
        """Test successful COBE data loading."""
        # Mock COBE data file content
        mock_data = """# COBE DMR Power Spectrum
# ell    C_ell_TT    sigma
2    100.0    10.0
3    150.0    15.0
4    200.0    20.0
5    180.0    18.0
"""
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = data_loader.load_cobe()
            
            assert isinstance(result, dict)
            assert 'TT' in result
            ell, C_ell, C_ell_err = result['TT']
            
            assert len(ell) == 4
            assert len(C_ell) == 4
            assert len(C_ell_err) == 4
            assert ell[0] == 2.0
            assert C_ell[0] == 100.0
            assert C_ell_err[0] == 10.0

    def test_load_cobe_cached(self, data_loader, tmp_path):
        """Test loading COBE data from cache."""
        cached_file = tmp_path / "downloaded_data" / "cobe_dmr_tt_spectrum.dat"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create cached data file
        cached_data = np.array([
            [2.0, 100.0, 10.0],
            [3.0, 150.0, 15.0],
            [4.0, 200.0, 20.0]
        ])
        np.savetxt(cached_file, cached_data, fmt='%.6e', header='ell C_ell_TT sigma')
        
        data_loader.use_cache = True
        result = data_loader.load_cobe()
        
        assert isinstance(result, dict)
        assert 'TT' in result
        ell, C_ell, C_ell_err = result['TT']
        
        assert len(ell) == 3
        assert ell[0] == 2.0
        assert C_ell[0] == 100.0

    def test_load_cobe_network_error(self, data_loader):
        """Test COBE loading fails hard on network error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_cobe()
            
            assert "Network error" in str(exc_info.value)

    def test_load_cobe_no_data_points(self, data_loader):
        """Test COBE loading fails hard when no valid data points found."""
        mock_data = "# Header only\n# No data\n"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_cobe()
            
            assert "No valid data points" in str(exc_info.value)

    def test_load_cobe_filters_multipoles(self, data_loader):
        """Test COBE loading filters multipoles outside valid range."""
        mock_data = """# COBE DMR Power Spectrum
# ell    C_ell_TT    sigma
1    50.0    5.0      # Below minimum (should be filtered)
2    100.0    10.0    # Valid
3    150.0    15.0    # Valid
4    200.0    20.0    # Valid
101  50.0    5.0      # Above maximum (should be filtered)
"""
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = data_loader.load_cobe()
            
            assert isinstance(result, dict)
            assert 'TT' in result
            ell, C_ell, C_ell_err = result['TT']
            
            # Should only have 3 valid points (ell=2, 3, 4)
            assert len(ell) == 3
            assert all(2 <= e <= 100 for e in ell)


class TestWMAPLoader:
    """Test WMAP data loading."""

    @pytest.fixture
    def data_loader(self, tmp_path):
        """Create a DataLoader instance with temporary directories."""
        return DataLoader(
            downloaded_data_dir=str(tmp_path / "downloaded_data"),
            processed_data_dir=str(tmp_path / "processed_data"),
            use_cache=False
        )

    def test_load_wmap_success(self, data_loader):
        """Test successful WMAP data loading."""
        # Mock WMAP data file content (C_ell format)
        mock_data_tt = """# WMAP 9-year Power Spectrum TT
# ell    C_ell_TT    sigma
2    100.0    10.0
3    150.0    15.0
4    200.0    20.0
5    180.0    18.0
"""
        mock_data_te = """# WMAP 9-year Power Spectrum TE
# ell    C_ell_TE    sigma
2    50.0    5.0
3    75.0    7.5
4    100.0    10.0
"""
        
        with patch('requests.get') as mock_get:
            def side_effect(url, **kwargs):
                mock_response = Mock()
                if 'tt' in url.lower():
                    mock_response.text = mock_data_tt
                elif 'te' in url.lower():
                    mock_response.text = mock_data_te
                else:
                    mock_response.text = ""
                mock_response.raise_for_status = Mock()
                return mock_response
            
            mock_get.side_effect = side_effect
            
            result = data_loader.load_wmap()
            
            assert isinstance(result, dict)
            assert 'TT' in result
            ell, C_ell, C_ell_err = result['TT']
            
            assert len(ell) == 4
            assert len(C_ell) == 4
            assert len(C_ell_err) == 4
            assert ell[0] == 2.0
            assert C_ell[0] == 100.0
            assert C_ell_err[0] == 10.0
            
            # TE should also be loaded if available
            if 'TE' in result:
                ell_te, C_ell_te, C_ell_err_te = result['TE']
                assert len(ell_te) == 4  # Mock data has 4 points

    def test_load_wmap_d_ell_format(self, data_loader):
        """Test WMAP loading with D_ell format (converts to C_ell)."""
        # Mock WMAP data file content (D_ell format, values > 1000)
        mock_data = """# WMAP 9-year Power Spectrum (D_ell format)
# ell    D_ell_TT    sigma
2    1256.637    125.664
3    1884.956    188.496
4    2513.274    251.327
"""
        
        with patch('requests.get') as mock_get:
            def side_effect(url, **kwargs):
                mock_response = Mock()
                if 'tt' in url.lower():
                    mock_response.text = mock_data
                else:
                    mock_response.text = ""
                mock_response.raise_for_status = Mock()
                return mock_response
            
            mock_get.side_effect = side_effect
            
            result = data_loader.load_wmap()
            
            assert isinstance(result, dict)
            assert 'TT' in result
            ell, C_ell, C_ell_err = result['TT']
            
            assert len(ell) == 3
            # D_ell = ell(ell+1)C_ell/(2pi), so C_ell = D_ell * 2pi / (ell(ell+1))
            # For ell=2: C_ell = 1256.637 * 2pi / (2*3) = 1256.637 * 6.283 / 6 ≈ 1315.9
            # Verify conversion happened: check that values were converted from D_ell format
            expected_c_ell_0 = 1256.637 * (2 * np.pi) / (2 * 3)
            assert abs(C_ell[0] - expected_c_ell_0) < 1.0, f"C_ell[0]={C_ell[0]}, expected≈{expected_c_ell_0}, all C_ell={C_ell}"
            # Verify that conversion formula was applied (values should match D_ell * 2pi / (ell(ell+1)))
            for i, ell_val in enumerate(ell):
                d_ell_val = [1256.637, 1884.956, 2513.274][i]
                expected_c_ell = d_ell_val * (2 * np.pi) / (ell_val * (ell_val + 1))
                assert abs(C_ell[i] - expected_c_ell) < 1.0, f"C_ell[{i}]={C_ell[i]}, expected≈{expected_c_ell}"

    def test_load_wmap_cached(self, data_loader, tmp_path):
        """Test loading WMAP data from cache."""
        cached_dir = tmp_path / "downloaded_data" / "wmap"
        cached_dir.mkdir(parents=True, exist_ok=True)
        
        cached_file_tt = cached_dir / "wmap_tt_spectrum.dat"
        cached_file_te = cached_dir / "wmap_te_spectrum.dat"
        
        # Create cached data files
        cached_data = np.array([
            [2.0, 100.0, 10.0],
            [3.0, 150.0, 15.0],
            [4.0, 200.0, 20.0]
        ])
        np.savetxt(cached_file_tt, cached_data, fmt='%.6e', header='ell C_ell_TT sigma')
        np.savetxt(cached_file_te, cached_data, fmt='%.6e', header='ell C_ell_TE sigma')
        
        data_loader.use_cache = True
        result = data_loader.load_wmap()
        
        assert isinstance(result, dict)
        assert 'TT' in result
        ell, C_ell, C_ell_err = result['TT']
        
        assert len(ell) == 3
        assert ell[0] == 2.0
        assert C_ell[0] == 100.0
        
        # TE should also be loaded if cached
        if 'TE' in result:
            ell_te, C_ell_te, C_ell_err_te = result['TE']
            assert len(ell_te) == 3

    def test_load_wmap_network_error(self, data_loader):
        """Test WMAP loading fails hard on network error."""
        with patch('requests.get') as mock_get:
            # First call (TT) fails, second call (TE) also fails
            mock_get.side_effect = requests.RequestException("Network error")
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_wmap()
            
            assert "No WMAP data available" in str(exc_info.value) or "Network error" in str(exc_info.value)

    def test_load_wmap_no_data_points(self, data_loader):
        """Test WMAP loading fails hard when no valid data points found."""
        mock_data = "# Header only\n# No data\n"
        
        with patch('requests.get') as mock_get:
            def side_effect(url):
                mock_response = Mock()
                mock_response.text = mock_data
                mock_response.raise_for_status = Mock()
                return mock_response
            
            mock_get.side_effect = side_effect
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_wmap()
            
            assert "No WMAP data available" in str(exc_info.value) or "No valid data points" in str(exc_info.value)

    def test_load_wmap_filters_multipoles(self, data_loader):
        """Test WMAP loading filters multipoles outside valid range."""
        mock_data = """# WMAP 9-year Power Spectrum
# ell    C_ell_TT    sigma
1    50.0    5.0      # Below minimum (should be filtered)
2    100.0    10.0    # Valid
3    150.0    15.0    # Valid
4    200.0    20.0    # Valid
2001  50.0    5.0     # Above maximum (should be filtered)
"""
        
        with patch('requests.get') as mock_get:
            def side_effect(url, **kwargs):
                mock_response = Mock()
                if 'tt' in url.lower():
                    mock_response.text = mock_data
                else:
                    mock_response.text = ""
                mock_response.raise_for_status = Mock()
                return mock_response
            
            mock_get.side_effect = side_effect
            
            result = data_loader.load_wmap()
            
            assert isinstance(result, dict)
            assert 'TT' in result
            ell, C_ell, C_ell_err = result['TT']
            
            # Should only have 3 valid points (ell=2, 3, 4)
            assert len(ell) == 3
            assert all(2 <= e <= 2000 for e in ell)


class TestCMBLoaderIntegration:
    """Integration tests for CMB data loading in ML pipeline context."""

    @pytest.fixture
    def data_loader(self, tmp_path):
        """Create a DataLoader instance with temporary directories."""
        return DataLoader(
            downloaded_data_dir=str(tmp_path / "downloaded_data"),
            processed_data_dir=str(tmp_path / "processed_data"),
            use_cache=False
        )

    def test_cobe_wmap_in_ml_pipeline(self, tmp_path):
        """Test that COBE and WMAP can be loaded for ML pipeline."""
        from pipeline.ml.ml_pipeline import MLPipeline
        
        # Mock data loader methods - now return dicts
        with patch('data.loader.DataLoader.load_cobe') as mock_cobe, \
             patch('data.loader.DataLoader.load_wmap') as mock_wmap, \
             patch('data.loader.DataLoader.load_act_dr6') as mock_act, \
             patch('data.loader.DataLoader.load_planck_2018') as mock_planck, \
             patch('data.loader.DataLoader.load_spt3g') as mock_spt3g:
            
            mock_cobe.return_value = {
                'TT': (np.array([2, 3, 4]), np.array([100, 150, 200]), np.array([10, 15, 20]))
            }
            mock_wmap.return_value = {
                'TT': (np.array([2, 3, 4]), np.array([100, 150, 200]), np.array([10, 15, 20])),
                'TE': (np.array([2, 3, 4]), np.array([50, 75, 100]), np.array([5, 7.5, 10]))
            }
            mock_act.return_value = {
                'TT': (np.array([2, 3, 4]), np.array([100, 150, 200]), np.array([10, 15, 20])),
                'TE': (np.array([2, 3, 4]), np.array([50, 75, 100]), np.array([5, 7.5, 10])),
                'EE': (np.array([2, 3, 4]), np.array([25, 37.5, 50]), np.array([2.5, 3.75, 5]))
            }
            mock_planck.return_value = {
                'TT': (np.array([2, 3, 4]), np.array([100, 150, 200]), np.array([10, 15, 20])),
                'TE': (np.array([2, 3, 4]), np.array([50, 75, 100]), np.array([5, 7.5, 10])),
                'EE': (np.array([2, 3, 4]), np.array([25, 37.5, 50]), np.array([2.5, 3.75, 5]))
            }
            mock_spt3g.return_value = {
                'TT': (np.array([2, 3, 4]), np.array([100, 150, 200]), np.array([10, 15, 20])),
                'TE': (np.array([2, 3, 4]), np.array([50, 75, 100]), np.array([5, 7.5, 10])),
                'EE': (np.array([2, 3, 4]), np.array([25, 37.5, 50]), np.array([2.5, 3.75, 5]))
            }
            
            pipeline = MLPipeline(output_dir=str(tmp_path / "results"))
            pipeline.context = {'dataset': 'cmb'}
            
            # This should not raise an error
            data = pipeline._load_all_cosmological_data()
            
            # Check that COBE TT and WMAP TT/TE are in the data
            assert 'cmb_cobe_tt' in data
            assert 'cmb_wmap_tt' in data
            assert 'cmb_wmap_te' in data
            assert data['cmb_cobe_tt']['source'] == 'COBE DMR TT'
            assert data['cmb_wmap_tt']['source'] == 'WMAP TT'
            assert data['cmb_wmap_te']['source'] == 'WMAP TE'
            
            # Check that ACT DR6, Planck, and SPT-3G have TT/TE/EE
            assert 'cmb_act_dr6_tt' in data
            assert 'cmb_act_dr6_te' in data
            assert 'cmb_act_dr6_ee' in data
            assert 'cmb_planck_2018_tt' in data
            assert 'cmb_planck_2018_te' in data
            assert 'cmb_planck_2018_ee' in data
            assert 'cmb_spt3g_tt' in data
            assert 'cmb_spt3g_te' in data
            assert 'cmb_spt3g_ee' in data

