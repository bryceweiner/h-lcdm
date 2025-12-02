"""
Unit tests for gravitational wave data loading.

Tests loading of LIGO, Virgo, and KAGRA GW event catalogs from GWOSC.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import requests

from data.loader import DataLoader, DataUnavailableError


class TestGWLoader:
    """Test gravitational wave data loading."""

    @pytest.fixture
    def data_loader(self, tmp_path):
        """Create a DataLoader instance with temporary directories."""
        return DataLoader(
            downloaded_data_dir=str(tmp_path / "downloaded_data"),
            processed_data_dir=str(tmp_path / "processed_data"),
            use_cache=False
        )

    def test_load_gw_data_ligo_success(self, data_loader):
        """Test successful LIGO GW data loading."""
        # Mock GWOSC API response
        mock_response_data = {
            'events': {
                'GW150914': {
                    'GPS': 1126259462.4,
                    'mass_1': {'median': 35.6},
                    'mass_2': {'median': 30.6},
                    'chirp_mass': {'median': 28.3},
                    'luminosity_distance': {'median': 410.0},
                    'redshift': {'median': 0.09},
                    'snr': 24.4,
                    'far': 1e-23,
                    'run': 'O1',
                    'network': 'LIGO'
                },
                'GW151226': {
                    'GPS': 1135136350.6,
                    'mass_1': {'median': 14.2},
                    'mass_2': {'median': 7.5},
                    'chirp_mass': {'median': 8.9},
                    'luminosity_distance': {'median': 440.0},
                    'redshift': {'median': 0.09},
                    'snr': 13.0,
                    'far': 1e-20,
                    'run': 'O1',
                    'network': 'LIGO'
                }
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            df = data_loader.load_gw_data(detector='ligo', run='O1')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'event_name' in df.columns
            assert 'detector' in df.columns
            assert 'mass_1' in df.columns
            assert 'mass_2' in df.columns
            assert 'chirp_mass' in df.columns
            assert df['detector'].iloc[0] == 'LIGO'
            assert df['run'].iloc[0] == 'O1'

    def test_load_gw_data_virgo_success(self, data_loader):
        """Test successful Virgo GW data loading."""
        mock_response_data = {
            'events': {
                'GW170814': {
                    'GPS': 1186741861.5,
                    'mass_1': {'median': 30.7},
                    'mass_2': {'median': 25.3},
                    'chirp_mass': {'median': 24.1},
                    'luminosity_distance': {'median': 540.0},
                    'redshift': {'median': 0.11},
                    'snr': 10.2,
                    'far': 1e-18,
                    'run': 'O2',
                    'network': 'LIGO-Virgo'
                }
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            df = data_loader.load_gw_data(detector='virgo', run='O2')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert 'VIRGO' in df['network'].iloc[0] or 'Virgo' in df['network'].iloc[0]

    def test_load_gw_data_kagra_success(self, data_loader):
        """Test successful KAGRA GW data loading."""
        mock_response_data = {
            'events': {
                'GW200311_115853': {
                    'GPS': 1266880733.0,
                    'mass_1': {'median': 28.3},
                    'mass_2': {'median': 24.6},
                    'chirp_mass': {'median': 22.1},
                    'luminosity_distance': {'median': 380.0},
                    'redshift': {'median': 0.08},
                    'snr': 12.5,
                    'far': 1e-19,
                    'run': 'O3',
                    'network': 'LIGO-Virgo-KAGRA'
                }
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            df = data_loader.load_gw_data(detector='kagra', run='O3')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert 'KAGRA' in df['network'].iloc[0] or 'Kagra' in df['network'].iloc[0]

    def test_load_gw_data_all_runs(self, data_loader):
        """Test loading GW data from all runs."""
        mock_response_data = {
            'events': {
                'GW150914': {
                    'GPS': 1126259462.4,
                    'mass_1': {'median': 35.6},
                    'mass_2': {'median': 30.6},
                    'chirp_mass': {'median': 28.3},
                    'luminosity_distance': {'median': 410.0},
                    'redshift': {'median': 0.09},
                    'snr': 24.4,
                    'far': 1e-23,
                    'run': 'O1',
                    'network': 'LIGO'
                }
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            df = data_loader.load_gw_data(detector='ligo', run=None)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 1

    def test_load_gw_data_cached(self, data_loader, tmp_path):
        """Test loading GW data from cache."""
        cached_file = tmp_path / "downloaded_data" / "gw_ligo_o1.pkl"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create cached data
        cached_df = pd.DataFrame({
            'event_name': ['GW150914'],
            'detector': ['LIGO'],
            'run': ['O1'],
            'mass_1': [35.6],
            'mass_2': [30.6],
            'chirp_mass': [28.3],
            'luminosity_distance': [410.0],
            'redshift': [0.09],
            'snr': [24.4],
            'far': [1e-23],
            'network': ['LIGO']
        })
        cached_df.to_pickle(cached_file)
        
        data_loader.use_cache = True
        df = data_loader.load_gw_data(detector='ligo', run='O1')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['event_name'].iloc[0] == 'GW150914'

    def test_load_gw_data_network_error(self, data_loader):
        """Test GW loading fails hard on network error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_gw_data(detector='ligo', run='O1')
            
            assert "Network error" in str(exc_info.value)

    def test_load_gw_data_no_events(self, data_loader):
        """Test GW loading fails hard when no events found."""
        mock_response_data = {'events': {}}
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_gw_data(detector='ligo', run='O1')
            
            assert "No events found" in str(exc_info.value)

    def test_load_gw_data_all_detectors(self, data_loader):
        """Test loading GW data from all detectors."""
        mock_response_data = {
            'events': {
                'GW150914': {
                    'GPS': 1126259462.4,
                    'mass_1': {'median': 35.6},
                    'mass_2': {'median': 30.6},
                    'chirp_mass': {'median': 28.3},
                    'luminosity_distance': {'median': 410.0},
                    'redshift': {'median': 0.09},
                    'snr': 24.4,
                    'far': 1e-23,
                    'run': 'O1',
                    'network': 'LIGO'
                }
            }
        }
        
        with patch('data.loader.DataLoader.load_gw_data') as mock_load:
            mock_load.side_effect = [
                pd.DataFrame({'event_name': ['GW150914'], 'detector': ['LIGO']}),
                pd.DataFrame({'event_name': ['GW170814'], 'detector': ['Virgo']}),
                pd.DataFrame({'event_name': ['GW200311'], 'detector': ['KAGRA']})
            ]
            
            results = data_loader.load_gw_data_all_detectors(run='O1')
            
            assert isinstance(results, dict)
            assert 'ligo' in results
            assert 'virgo' in results
            assert 'kagra' in results
            assert isinstance(results['ligo'], pd.DataFrame)
            assert isinstance(results['virgo'], pd.DataFrame)
            assert isinstance(results['kagra'], pd.DataFrame)

    def test_load_gw_data_all_detectors_no_data(self, data_loader):
        """Test loading all detectors fails hard when no data available."""
        with patch('data.loader.DataLoader.load_gw_data') as mock_load:
            mock_load.side_effect = DataUnavailableError("No data")
            
            with pytest.raises(DataUnavailableError) as exc_info:
                data_loader.load_gw_data_all_detectors(run='O1')
            
            assert "No GW data available" in str(exc_info.value)


class TestGWLoaderIntegration:
    """Integration tests for GW data loading in ML pipeline context."""

    @pytest.fixture
    def data_loader(self, tmp_path):
        """Create a DataLoader instance with temporary directories."""
        return DataLoader(
            downloaded_data_dir=str(tmp_path / "downloaded_data"),
            processed_data_dir=str(tmp_path / "processed_data"),
            use_cache=False
        )

    def test_gw_data_in_ml_pipeline(self, tmp_path):
        """Test that GW data can be loaded for ML pipeline."""
        from pipeline.ml.ml_pipeline import MLPipeline
        
        # Mock GW data loader methods
        with patch('data.loader.DataLoader.load_gw_data_all_detectors') as mock_gw:
            mock_gw.return_value = {
                'ligo': pd.DataFrame({
                    'event_name': ['GW150914'],
                    'detector': ['LIGO'],
                    'mass_1': [35.6],
                    'mass_2': [30.6],
                    'chirp_mass': [28.3],
                    'luminosity_distance': [410.0],
                    'redshift': [0.09],
                    'snr': [24.4],
                    'far': [1e-23]
                }),
                'virgo': pd.DataFrame({
                    'event_name': ['GW170814'],
                    'detector': ['Virgo'],
                    'mass_1': [30.7],
                    'mass_2': [25.3],
                    'chirp_mass': [24.1],
                    'luminosity_distance': [540.0],
                    'redshift': [0.11],
                    'snr': [10.2],
                    'far': [1e-18]
                }),
                'kagra': pd.DataFrame({
                    'event_name': ['GW200311'],
                    'detector': ['KAGRA'],
                    'mass_1': [28.3],
                    'mass_2': [24.6],
                    'chirp_mass': [22.1],
                    'luminosity_distance': [380.0],
                    'redshift': [0.08],
                    'snr': [12.5],
                    'far': [1e-19]
                })
            }
            
            # Mock other data loaders
            with patch('data.loader.DataLoader.load_frb_data') as mock_frb, \
                 patch('data.loader.DataLoader.load_lyman_alpha_data') as mock_lyman, \
                 patch('data.loader.DataLoader.load_jwst_data') as mock_jwst:
                
                mock_frb.return_value = pd.DataFrame({'ra': [0.0], 'dec': [0.0]})
                mock_lyman.return_value = pd.DataFrame({'wavelength': [1216.0], 'flux': [1.0]})
                mock_jwst.return_value = pd.DataFrame({'ra': [0.0], 'dec': [0.0], 'z': [8.0]})
                
                pipeline = MLPipeline(output_dir=str(tmp_path / "results"))
                pipeline.context = {'dataset': 'all'}
                
                # This should not raise an error
                data = pipeline._load_all_cosmological_data()
                
                # Check that GW data is in the data
                assert 'gw_ligo' in data
                assert 'gw_virgo' in data
                assert 'gw_kagra' in data
                assert isinstance(data['gw_ligo'], pd.DataFrame)
                assert isinstance(data['gw_virgo'], pd.DataFrame)
                assert isinstance(data['gw_kagra'], pd.DataFrame)

