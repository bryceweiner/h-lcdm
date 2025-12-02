"""
Data Preparation Module
=======================

Handles all data loading and preparation for the ML pipeline.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from data.loader import DataLoader, DataUnavailableError


class DataPreparation:
    """
    Handles data loading and preparation for ML pipeline.
    
    Separated from main pipeline for better organization and testability.
    """
    
    def __init__(self, data_loader: DataLoader, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        """
        Initialize data preparation.
        
        Parameters:
            data_loader: DataLoader instance
            logger: Logger instance
            context: Execution context (for dataset preference)
        """
        self.data_loader = data_loader
        self.logger = logger
        self.context = context or {}
    
    def load_all_cosmological_data(self) -> Dict[str, Any]:
        """Load all available cosmological data for training."""
        data = {}

        # Get dataset preference from context
        dataset_pref = self.context.get('dataset', 'all')

        if dataset_pref == 'cmb':
            # CMB-only mode: Load multiple CMB datasets
            self.logger.info("Loading CMB data (ACT DR6, Planck 2018, SPT-3G, COBE, WMAP)...")
            data = self._load_cmb_data(data)
            
            # Check if any CMB data was loaded
            cmb_keys = [k for k in data.keys() if k.startswith('cmb_')]
            if not cmb_keys:
                raise DataUnavailableError("No CMB data available")
                
        elif dataset_pref == 'galaxy':
            # Galaxy-only mode
            self.logger.info("Loading galaxy catalog data...")
            try:
                galaxy_data = self.data_loader.load_sdss_galaxy_catalog()
                data['galaxy'] = galaxy_data
            except Exception as e:
                raise DataUnavailableError(f"Galaxy catalog loading failed: {e}")
                
        else:
            # 'all' mode: Load all available data
            data = self._load_cmb_data(data)
            data = self._load_bao_data(data)
            data = self._load_void_data(data)
            data = self._load_other_modalities(data)

        return data
    
    def _load_cmb_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load CMB data (ACT DR6, Planck 2018, SPT-3G, COBE, WMAP)."""
        self.logger.info("Loading CMB data (ACT DR6, Planck 2018, SPT-3G, COBE, WMAP)...")
        
        # Load ACT DR6 (returns dict with TT, TE, EE)
        try:
            act_result = self.data_loader.load_act_dr6()
            if act_result:
                for spectra in ['TT', 'TE', 'EE']:
                    if spectra in act_result:
                        ell, C_ell, C_ell_err = act_result[spectra]
                        data[f'cmb_act_dr6_{spectra.lower()}'] = {
                            'ell': ell,
                            'C_ell': C_ell,
                            'C_ell_err': C_ell_err,
                            'source': f'ACT DR6 {spectra}'
                        }
                        self.logger.info(f"✓ Loaded ACT DR6 {spectra}: {len(ell)} multipoles")
        except Exception as e:
            self.logger.warning(f"ACT DR6 loading failed: {e}")
        
        # Load Planck 2018 (returns dict with TT, TE, EE)
        try:
            planck_result = self.data_loader.load_planck_2018()
            if planck_result:
                for spectra in ['TT', 'TE', 'EE']:
                    if spectra in planck_result:
                        ell, C_ell, C_ell_err = planck_result[spectra]
                        data[f'cmb_planck_2018_{spectra.lower()}'] = {
                            'ell': ell,
                            'C_ell': C_ell,
                            'C_ell_err': C_ell_err,
                            'source': f'Planck 2018 {spectra}'
                        }
                        self.logger.info(f"✓ Loaded Planck 2018 {spectra}: {len(ell)} multipoles")
        except Exception as e:
            self.logger.warning(f"Planck 2018 loading failed: {e}")
        
        # Load SPT-3G (returns dict with TT, TE, EE)
        try:
            spt3g_result = self.data_loader.load_spt3g()
            if spt3g_result:
                for spectra in ['TT', 'TE', 'EE']:
                    if spectra in spt3g_result:
                        ell, C_ell, C_ell_err = spt3g_result[spectra]
                        data[f'cmb_spt3g_{spectra.lower()}'] = {
                            'ell': ell,
                            'C_ell': C_ell,
                            'C_ell_err': C_ell_err,
                            'source': f'SPT-3G {spectra}'
                        }
                        self.logger.info(f"✓ Loaded SPT-3G {spectra}: {len(ell)} multipoles")
        except Exception as e:
            self.logger.warning(f"SPT-3G loading failed: {e}")
        
        # Load COBE (returns dict with TT)
        try:
            cobe_result = self.data_loader.load_cobe()
            if cobe_result and 'TT' in cobe_result:
                ell, C_ell, C_ell_err = cobe_result['TT']
                data['cmb_cobe_tt'] = {
                    'ell': ell,
                    'C_ell': C_ell,
                    'C_ell_err': C_ell_err,
                    'source': 'COBE DMR TT'
                }
                self.logger.info(f"✓ Loaded COBE TT: {len(ell)} multipoles")
        except Exception as e:
            self.logger.warning(f"COBE loading failed: {e}")
        
        # Load WMAP (returns dict with TT, TE)
        try:
            wmap_result = self.data_loader.load_wmap()
            if wmap_result:
                for spectra in ['TT', 'TE']:
                    if spectra in wmap_result:
                        ell, C_ell, C_ell_err = wmap_result[spectra]
                        data[f'cmb_wmap_{spectra.lower()}'] = {
                            'ell': ell,
                            'C_ell': C_ell,
                            'C_ell_err': C_ell_err,
                            'source': f'WMAP {spectra}'
                        }
                        self.logger.info(f"✓ Loaded WMAP {spectra}: {len(ell)} multipoles")
        except Exception as e:
            self.logger.warning(f"WMAP loading failed: {e}")
        
        return data
    
    def _load_bao_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load BAO data (BOSS DR12, DESI, eBOSS)."""
        self.logger.info("Loading BAO data (BOSS DR12, DESI, eBOSS)...")
        bao_surveys = ['boss_dr12', 'desi', 'eboss']
        for survey in bao_surveys:
            try:
                bao_data = self.data_loader.load_bao_data(survey=survey)
                if bao_data:
                    data[f'bao_{survey}'] = bao_data
                    self.logger.info(f"✓ Loaded BAO {survey}")
            except Exception as e:
                self.logger.warning(f"BAO {survey} loading failed: {e}")
        return data
    
    def _load_void_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load void catalog data."""
        # Void data - check for deduplicated files first, skip processing if they exist
        deduplicated_path = Path(self.data_loader.processed_data_dir) / "voids_deduplicated.pkl"
        hlcdm_deduplicated_path = Path(self.data_loader.processed_data_dir) / "voids_hlcdm_deduplicated.pkl"
        
        if deduplicated_path.exists() and hlcdm_deduplicated_path.exists():
            self.logger.info("Found deduplicated void catalogs, loading from pickle files (skipping void pipeline processing)...")
            try:
                import pickle
                with open(deduplicated_path, 'rb') as f:
                    voids_deduplicated = pickle.load(f)
                with open(hlcdm_deduplicated_path, 'rb') as f:
                    voids_hlcdm_deduplicated = pickle.load(f)
                
                if voids_deduplicated is not None and not voids_deduplicated.empty:
                    data['void_sdss_dr7'] = voids_deduplicated[voids_deduplicated['survey'] == 'SDSS_DR7'].copy()
                    data['void_sdss_dr16'] = voids_deduplicated[voids_deduplicated['survey'] == 'SDSS_DR16'].copy()
                    self.logger.info(f"✓ Loaded deduplicated void catalogs: {len(voids_deduplicated)} total voids")
                
                if len(voids_hlcdm_deduplicated) > 0:
                    self.logger.info(f"✓ Loaded H-ΛCDM deduplicated void catalog: {len(voids_hlcdm_deduplicated)} voids")
            except Exception as e:
                raise ValueError(
                    f"CRITICAL ERROR: Failed to load deduplicated void catalogs: {e}. "
                    f"Fix the pickle files or regenerate them using the void pipeline."
                ) from e
        else:
            # Deduplicated files don't exist - run void pipeline processing
            self.logger.info("Deduplicated void catalogs not found, running void pipeline processing...")
            # Try SDSS DR7 voids
            try:
                from data.processors.void_processor import VoidDataProcessor
                void_processor = VoidDataProcessor()
                processed = void_processor.process_void_catalogs(surveys=['sdss_dr7_douglass'], force_reprocess=False)
                if processed and 'sdss_dr7_douglass' in processed:
                    void_dr7 = processed['sdss_dr7_douglass']
                    if void_dr7 is not None and not void_dr7.empty:
                        data['void_sdss_dr7'] = void_dr7
                        self.logger.info(f"✓ Loaded SDSS DR7 voids: {len(void_dr7)} voids")
            except Exception as e:
                raise ValueError(f"CRITICAL ERROR: SDSS DR7 void catalog loading failed: {e}") from e
            
            # Try SDSS DR16 voids
            try:
                from data.processors.void_processor import VoidDataProcessor
                void_processor = VoidDataProcessor()
                processed = void_processor.process_void_catalogs(surveys=['sdss_dr16_hzobov'], force_reprocess=False)
                if processed and 'sdss_dr16_hzobov' in processed:
                    void_dr16 = processed['sdss_dr16_hzobov']
                    if void_dr16 is not None and not void_dr16.empty:
                        data['void_sdss_dr16'] = void_dr16
                        self.logger.info(f"✓ Loaded SDSS DR16 voids: {len(void_dr16)} voids")
            except Exception as e:
                raise ValueError(f"CRITICAL ERROR: SDSS DR16 void catalog loading failed: {e}") from e
            
            # Try DESIVAST voids
            try:
                desivast_voids = self.data_loader.load_voidfinder_catalog('desivast')
                if desivast_voids is not None and not desivast_voids.empty:
                    data['void_desivast'] = desivast_voids
                    self.logger.info(f"✓ Loaded DESIVAST voids: {len(desivast_voids)} voids")
            except Exception as e:
                raise ValueError(f"CRITICAL ERROR: DESIVAST void catalog loading failed: {e}") from e
            
            # Fallback: try generic void catalog
            if not any(k.startswith('void_') for k in data.keys()):
                try:
                    void_data = self.data_loader.load_void_catalog()
                    if void_data is not None and not void_data.empty:
                        data['void'] = void_data
                        self.logger.info(f"✓ Loaded generic void catalog: {len(void_data)} voids")
                except Exception as e:
                    raise ValueError(f"CRITICAL ERROR: Generic void catalog loading failed: {e}") from e
        
        return data
    
    def _load_other_modalities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load other modalities (galaxy, FRB, Lyman-alpha, JWST, GW)."""
        # Galaxy data - required for 'all' mode, fail hard if unavailable
        galaxy_data = self.data_loader.load_sdss_galaxy_catalog()
        data['galaxy'] = galaxy_data

        # FRB data - required for 'all' mode, fail hard if unavailable
        frb_data = self.data_loader.load_frb_data()
        data['frb'] = frb_data

        # Lyman-alpha data - required for 'all' mode, fail hard if unavailable
        lyman_data = self.data_loader.load_lyman_alpha_data()
        data['lyman_alpha'] = lyman_data

        # JWST data - required for 'all' mode, fail hard if unavailable
        jwst_data = self.data_loader.load_jwst_data()
        data['jwst'] = jwst_data
        
        # Gravitational wave data - required for 'all' mode, fail hard if unavailable
        # Load from all detectors (LIGO, Virgo, KAGRA) across all runs
        try:
            gw_data_all = self.data_loader.load_gw_data_all_detectors()
            # Store each detector as separate modality
            for detector, gw_df in gw_data_all.items():
                if gw_df is not None and not gw_df.empty:
                    data[f'gw_{detector}'] = gw_df
                    self.logger.info(f"✓ Loaded GW {detector.upper()}: {len(gw_df)} events")
        except Exception as e:
            raise DataUnavailableError(f"GW data loading failed: {e}")
        
        return data
    
    def get_encoder_dimensions(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Get input dimensions for each modality encoder."""
        dims = {}

        # Placeholder dimensions based on typical data sizes
        modality_dims = {
            'cmb': 500,      # Power spectrum length
            'bao': 10,       # BAO measurements
            'bao_boss_dr12': 10,  # BOSS DR12 BAO
            'bao_desi': 10,  # DESI BAO
            'bao_eboss': 10,  # eBOSS BAO
            'void': 20,      # Void properties
            'void_sdss_dr7': 20,  # SDSS DR7 voids
            'void_sdss_dr16': 20,  # SDSS DR16 voids
            'void_desivast': 20,  # DESIVAST voids
            'galaxy': 30,    # Galaxy features
            'frb': 15,       # FRB properties
            'lyman_alpha': 100,  # Spectrum length
            'jwst': 25,      # JWST features
            'gw': 20,        # GW event parameters
            'gw_ligo': 20,   # LIGO GW events
            'gw_virgo': 20,  # Virgo GW events
            'gw_kagra': 20   # KAGRA GW events
        }

        # Handle CMB TT/TE/EE modalities dynamically
        for modality in data.keys():
            if modality.startswith('cmb_') and (modality.endswith('_tt') or modality.endswith('_te') or modality.endswith('_ee')):
                if data[modality] is not None:
                    dims[modality] = 500
        
        # Handle BAO sub-modalities
        for bao_survey in ['bao_boss_dr12', 'bao_desi', 'bao_eboss']:
            if bao_survey in data and data[bao_survey] is not None:
                dims[bao_survey] = 10
        
        # Handle void sub-modalities
        for void_catalog in ['void_sdss_dr7', 'void_sdss_dr16', 'void_desivast']:
            if void_catalog in data and data[void_catalog] is not None:
                dims[void_catalog] = 20
        
        # Handle GW sub-modalities
        for gw_detector in ['gw_ligo', 'gw_virgo', 'gw_kagra']:
            if gw_detector in data and data[gw_detector] is not None:
                dims[gw_detector] = 20
        
        # Then process other modalities
        for modality in data.keys():
            if modality.startswith('cmb_'):
                continue  # Already handled above
            if modality in modality_dims and data[modality] is not None:
                dims[modality] = modality_dims[modality]
            elif modality.startswith('cmb') and data[modality] is not None:
                # Handle generic CMB if no sub-modalities
                dims[modality] = 500

        return dims
    
    def get_survey_mapping(self, cosmological_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Get mapping of survey names to their modalities.
        
        Parameters:
            cosmological_data: Dictionary of loaded cosmological data
        
        Returns:
            Dictionary mapping survey names to lists of modality keys
        """
        return {
            'ACT': [k for k in cosmological_data.keys() if k.startswith('cmb_act_dr6_')],
            'Planck': [k for k in cosmological_data.keys() if k.startswith('cmb_planck_2018_')],
            'SPT-3G': [k for k in cosmological_data.keys() if k.startswith('cmb_spt3g_')],
            'COBE': [k for k in cosmological_data.keys() if k.startswith('cmb_cobe_')],
            'WMAP': [k for k in cosmological_data.keys() if k.startswith('cmb_wmap_')],
            'BOSS': ['bao_boss_dr12'],
            'DESI': ['bao_desi', 'void_desivast'],
            'eBOSS': ['bao_eboss'],
            'SDSS': ['void_sdss_dr7', 'void_sdss_dr16', 'galaxy'],
            'FRB': ['frb'],
            'Lyman-alpha': ['lyman_alpha'],
            'JWST': ['jwst'],
            'LIGO': [k for k in cosmological_data.keys() if k.startswith('gw_ligo')],
            'Virgo': [k for k in cosmological_data.keys() if k.startswith('gw_virgo')],
            'KAGRA': [k for k in cosmological_data.keys() if k.startswith('gw_kagra')]
        }

