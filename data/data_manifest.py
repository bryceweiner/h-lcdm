"""
Data Manifest for ML Training
=============================

Tracks all data sources used in ML training, including known H-ΛCDM signals
that are BLINDED to the ML algorithms during training.

This manifest ensures reproducibility and tracks data provenance for the blind
ML testing approach where ML must detect H-ΛCDM signals without knowing they exist.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class DataSource:
    """Represents a single data source for ML training."""
    name: str
    modality: str  # 'cmb', 'bao', 'void', 'galaxy', 'frb', 'lyman_alpha', 'jwst'
    survey: str
    n_samples: int
    features: List[str]
    redshift_range: Optional[tuple] = None
    magnitude_range: Optional[tuple] = None
    data_quality: str = 'unknown'  # 'gold', 'silver', 'bronze'
    provenance: str = 'unknown'
    url: Optional[str] = None
    # BLINDED: Known H-ΛCDM signals present - NOT exposed to ML algorithms
    h_lcdm_signals_blinded: List[str] = None  # e.g., ['enhanced_sound_horizon', 'e8_alignment']


class DataManifest:
    """
    Unified manifest tracking all data sources for ML training.

    Maintains blind testing approach where ML detects patterns without
    knowing which data contains H-ΛCDM signals.
    """

    def __init__(self, manifest_file: str = "data_manifest.json"):
        """
        Initialize data manifest.

        Parameters:
            manifest_file: Path to save/load manifest
        """
        self.manifest_file = Path("data") / manifest_file
        self.sources: Dict[str, DataSource] = {}
        self.load_manifest()

    def add_source(self, source: DataSource):
        """Add a data source to the manifest."""
        self.sources[source.name] = source
        self.save_manifest()

    def get_sources_by_modality(self, modality: str) -> List[DataSource]:
        """Get all sources for a specific modality."""
        return [s for s in self.sources.values() if s.modality == modality]

    def get_all_modalities(self) -> List[str]:
        """Get list of all modalities in manifest."""
        return list(set(s.modality for s in self.sources.values()))

    def get_blinded_signals_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of blinded H-ΛCDM signals by modality.

        WARNING: This is for validation/reporting ONLY.
        ML algorithms MUST NOT access this information.
        """
        summary = {}
        for modality in self.get_all_modalities():
            sources = self.get_sources_by_modality(modality)
            signals = []
            for source in sources:
                if source.h_lcdm_signals_blinded:
                    signals.extend(source.h_lcdm_signals_blinded)
            summary[modality] = list(set(signals))  # Remove duplicates
        return summary

    def validate_ml_training_data(self) -> Dict[str, Any]:
        """
        Validate that data is suitable for ML training.

        Returns:
            dict: Validation results
        """
        validation = {
            'total_sources': len(self.sources),
            'modalities_covered': self.get_all_modalities(),
            'total_samples': sum(s.n_samples for s in self.sources.values()),
            'quality_distribution': {},
            'coverage_gaps': []
        }

        # Quality distribution
        for quality in ['gold', 'silver', 'bronze']:
            count = sum(1 for s in self.sources.values() if s.data_quality == quality)
            validation['quality_distribution'][quality] = count

        # Check for coverage gaps
        required_modalities = ['cmb', 'bao', 'void', 'galaxy', 'frb', 'lyman_alpha', 'jwst']
        missing_modalities = set(required_modalities) - set(validation['modalities_covered'])
        if missing_modalities:
            validation['coverage_gaps'].extend([f"Missing modality: {m}" for m in missing_modalities])

        # Check sample sizes
        small_sources = [s.name for s in self.sources.values() if s.n_samples < 100]
        if small_sources:
            validation['coverage_gaps'].append(f"Small datasets (<100 samples): {small_sources}")

        return validation

    def save_manifest(self):
        """Save manifest to JSON file."""
        data = {
            'sources': {
                name: {
                    'name': s.name,
                    'modality': s.modality,
                    'survey': s.survey,
                    'n_samples': s.n_samples,
                    'features': s.features,
                    'redshift_range': s.redshift_range,
                    'magnitude_range': s.magnitude_range,
                    'data_quality': s.data_quality,
                    'provenance': s.provenance,
                    'url': s.url,
                    # BLINDED signals saved for validation only
                    'h_lcdm_signals_blinded': s.h_lcdm_signals_blinded
                } for name, s in self.sources.items()
            }
        }

        with open(self.manifest_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_manifest(self):
        """Load manifest from JSON file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    data = json.load(f)

                for name, source_data in data.get('sources', {}).items():
                    source = DataSource(
                        name=source_data['name'],
                        modality=source_data['modality'],
                        survey=source_data['survey'],
                        n_samples=source_data['n_samples'],
                        features=source_data['features'],
                        redshift_range=tuple(source_data['redshift_range']) if source_data.get('redshift_range') else None,
                        magnitude_range=tuple(source_data['magnitude_range']) if source_data.get('magnitude_range') else None,
                        data_quality=source_data.get('data_quality', 'unknown'),
                        provenance=source_data.get('provenance', 'unknown'),
                        url=source_data.get('url'),
                        h_lcdm_signals_blinded=source_data.get('h_lcdm_signals_blinded')
                    )
                    self.sources[name] = source
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
                self.sources = {}

    def initialize_default_sources(self):
        """
        Initialize manifest with known H-ΛCDM data sources.

        This creates the blind testing setup where ML will be trained on
        data containing unknown H-ΛCDM signals.
        """
        # CMB data sources
        self.add_source(DataSource(
            name='act_dr6_ee',
            modality='cmb',
            survey='ACT',
            n_samples=1000,  # multipole bins
            features=['ell', 'C_ell', 'C_ell_err', 'bispectrum', 'topological_features'],
            data_quality='gold',
            provenance='published ACT DR6',
            url='https://lambda.gsfc.nasa.gov/product/act/act_dr6_lensing/',
            h_lcdm_signals_blinded=['phase_transitions_at_ell_1076_1706_2336']
        ))

        self.add_source(DataSource(
            name='planck_2018_ee',
            modality='cmb',
            survey='Planck',
            n_samples=800,
            features=['ell', 'C_ell', 'C_ell_err', 'bispectrum'],
            data_quality='gold',
            provenance='published Planck 2018',
            url='https://pla.esac.esa.int/',
            h_lcdm_signals_blinded=['phase_transitions_at_ell_1076_1706_2336']
        ))

        # BAO data sources
        self.add_source(DataSource(
            name='boss_dr12_bao',
            modality='bao',
            survey='BOSS DR12',
            n_samples=3,  # redshift bins
            features=['z', 'D_M_over_r_d', 'error', 'correlation_matrix'],
            redshift_range=(0.38, 0.61),
            data_quality='gold',
            provenance='published Alam et al. 2017',
            h_lcdm_signals_blinded=['enhanced_sound_horizon_rs_150.71']
        ))

        self.add_source(DataSource(
            name='desi_y1_bao',
            modality='bao',
            survey='DESI Year 1',
            n_samples=3,
            features=['z', 'D_M_over_r_d', 'error'],
            redshift_range=(0.51, 1.01),
            data_quality='silver',
            provenance='published DESI Collaboration 2024',
            h_lcdm_signals_blinded=['enhanced_sound_horizon_rs_150.71']
        ))

        # Void data sources (deduplicated from multiple surveys)
        self.add_source(DataSource(
            name='voids_deduplicated',
            modality='void',
            survey='SDSS DR7 + DESI DR1 + DES Y3 + 2MRS (combined)',
            n_samples=50000,  # estimated after deduplication
            features=['ra', 'dec', 'z', 'size', 'ellipticity', 'density_contrast', 'alignment_angles'],
            redshift_range=(0.005, 0.9),
            data_quality='gold',
            provenance='Douglass et al. 2023 + Clampitt & Jain 2015 + DESIVAST 2024 + DES Y3 + 2MRS',
            h_lcdm_signals_blinded=['e8_hetereotic_alignment_C_25_32']
        ))

        # Galaxy catalogs
        self.add_source(DataSource(
            name='sdss_dr16_galaxies',
            modality='galaxy',
            survey='SDSS DR16',
            n_samples=50000,
            features=['ra', 'dec', 'z', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag',
                     'petrosian_radius', 'concentration', 'u_minus_r'],
            redshift_range=(0.01, 1.0),
            magnitude_range=(15, 21),
            data_quality='gold',
            provenance='SDSS DR16 spectroscopic',
            h_lcdm_signals_blinded=['holographic_clustering']  # If present in SDSS scale
        ))

        # FRB data
        self.add_source(DataSource(
            name='frb_catalog',
            modality='frb',
            survey='Multiple telescopes',
            n_samples=200,
            features=['ra', 'dec', 'z', 'dm', 'timing_intervals', 'dispersion_measure'],
            data_quality='silver',
            provenance='Published FRB catalogs',
            h_lcdm_signals_blinded=['little_bang_information_saturation']
        ))

        # Lyman-alpha data
        self.add_source(DataSource(
            name='lyman_alpha_forest',
            modality='lyman_alpha',
            survey='Multiple quasar spectra',
            n_samples=1000,  # spectral segments
            features=['wavelength', 'flux', 'continuum', 'optical_depth', 'correlation_function'],
            redshift_range=(2.0, 4.0),
            data_quality='silver',
            provenance='Published quasar spectra',
            h_lcdm_signals_blinded=['phase_transitions_at_z_2.5']
        ))

        # JWST data
        self.add_source(DataSource(
            name='jwst_early_galaxies',
            modality='jwst',
            survey='JWST NIRCam',
            n_samples=100,
            features=['ra', 'dec', 'z', 'f200w_mag', 'f277w_mag', 'f356w_mag',
                     'half_light_radius', 'sersic_index', 'mass_estimate'],
            redshift_range=(8.0, 15.0),
            magnitude_range=(24, 30),
            data_quality='silver',
            provenance='Published JWST catalogs',
            h_lcdm_signals_blinded=['anti_viscosity_mass_limits']
        ))
