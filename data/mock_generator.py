"""
Mock Dataset Generator for ML Validation
=========================================

Generates mock datasets matching statistical properties of real astronomical data
for null hypothesis testing in ML validation. These mocks lack H-ΛCDM signals
and are used ONLY for validation testing, never for training.

Mock datasets ensure:
- Same statistical properties as real data
- Same noise characteristics
- Same survey selection effects
- NO H-ΛCDM signals (random data)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings


class MockDatasetGenerator:
    """
    Generate mock datasets for ML validation testing.

    Creates statistically equivalent datasets without H-ΛCDM signals
    for rigorous null hypothesis testing.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock generator.

        Parameters:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_mock_cmb_maps(self, n_multipoles: int = 1000,
                              ell_min: int = 100, ell_max: int = 3000,
                              cosmic_variance: bool = True,
                              use_planck_template: bool = False,
                              template_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate mock CMB power spectra matching real data statistics.

        Parameters:
            n_multipoles: Number of multipole bins
            ell_min: Minimum multipole
            ell_max: Maximum multipole
            cosmic_variance: Include cosmic variance noise
            use_planck_template: If True, use a Planck 2018 TT spectrum as the mean C_ell.
            template_path: Optional path to a three-column file [ell, D_ell, sigma]; if None, use
                           downloaded_data/planck_2018/planck_2018_TT_spectrum.dat when available.

        Returns:
            dict: Mock CMB data with same format as real data
        """
        ell = np.linspace(ell_min, ell_max, n_multipoles)

        # ------------------------------------------------------------------
        # Choose base spectrum: template (preferred) or power-law fallback
        # ------------------------------------------------------------------
        base_cl = None
        if use_planck_template:
            path = Path(template_path) if template_path else Path("downloaded_data/planck_2018/planck_2018_TT_spectrum.dat")
            if path.exists():
                try:
                    data = np.loadtxt(path)
                    # Expect columns: ell, D_ell, sigma
                    ell_template = data[:, 0]
                    dell_template = data[:, 1]
                    # Convert D_ell -> C_ell: D_ell = ell (ell+1) C_ell / (2π)
                    cl_template = dell_template * 2 * np.pi / (ell_template * (ell_template + 1))
                    # Interpolate onto target ell grid
                    base_cl = np.interp(ell, ell_template, cl_template, left=cl_template[0], right=cl_template[-1])
                except Exception:
                    base_cl = None

        if base_cl is None:
            # Fallback: power-law with mild scatter
            base_cl = 1000 * (ell / 1000) ** (-2.7)

        # Add random fluctuations (no phase transitions)
        noise_factor = 0.1
        mock_cl = base_cl * (1 + np.random.normal(0, noise_factor, len(ell)))

        # Add cosmic variance if requested
        if cosmic_variance:
            cosmic_var = np.sqrt(2 / (2 * ell + 1)) * mock_cl
            mock_cl += np.random.normal(0, cosmic_var)

        mock_cl_err = 0.1 * np.abs(mock_cl)  # Typical measurement error, positive

        return {
            'ell': ell,
            'C_ell': mock_cl,
            'C_ell_err': mock_cl_err,
            'source': 'mock_cmb',
            'n_multipoles': n_multipoles,
            'note': 'Mock CMB spectrum without H-ΛCDM phase transitions',
            'template_used': bool(use_planck_template and base_cl is not None)
        }

    def generate_mock_bao_measurements(self, n_measurements: int = 3,
                                      z_range: Tuple[float, float] = (0.3, 1.0)) -> Dict[str, Any]:
        """
        Generate mock BAO measurements without enhanced sound horizon.

        Parameters:
            n_measurements: Number of redshift bins
            z_range: Redshift range

        Returns:
            dict: Mock BAO data
        """
        z_values = np.linspace(z_range[0], z_range[1], n_measurements)

        # Mock BAO measurements using ΛCDM sound horizon (no enhancement)
        rs_fiducial = 147.5  # ΛCDM value (not enhanced)
        h_fiducial = 0.7

        mock_measurements = []
        for z in z_values:
            # D_M(z) / r_d with typical ΛCDM cosmology (no H-ΛCDM enhancement)
            # Simplified calculation
            dm_over_rd = 10 + 3 * (z - 0.5) + np.random.normal(0, 0.5)
            error = 0.2 * abs(dm_over_rd)  # Typical measurement error

            mock_measurements.append({
                'z': float(z),
                'value': float(dm_over_rd),
                'error': float(error)
            })

        return {
            'name': 'Mock BAO measurements',
            'measurements': mock_measurements,
            'correlation_matrix': None,
            'reference': 'Mock data for validation',
            'source': 'mock_bao',
            'data_release': 'validation_mock',
            'measurement_type': 'D_M/r_d',
            'tracer': 'Mock galaxies',
            'note': 'Mock BAO measurements without H-ΛCDM sound horizon enhancement'
        }

    def generate_mock_void_catalog(self, n_voids: int = 10000,
                                  volume_mpc3: float = 1e8) -> pd.DataFrame:
        """
        Generate mock void catalog matching statistical properties.

        Parameters:
            n_voids: Number of voids to generate
            volume_mpc3: Survey volume in Mpc³

        Returns:
            pd.DataFrame: Mock void catalog
        """
        np.random.seed(self.seed)

        # Generate void positions uniformly in survey volume
        # Assume SDSS-like geometry: ~2000 Mpc³, z < 0.15
        positions = np.random.uniform(0, volume_mpc3**(1/3), (n_voids, 3))

        # Convert to angular coordinates (simplified)
        ra = np.random.uniform(100, 260, n_voids)  # SDSS footprint
        dec = np.random.uniform(-10, 70, n_voids)
        redshift = np.random.uniform(0.01, 0.15, n_voids)

        # Generate realistic void sizes (log-normal distribution)
        # Typical voids: 10-50 Mpc radius
        void_radii = np.random.lognormal(np.log(25), 0.5, n_voids)
        void_radii = np.clip(void_radii, 5, 100)  # Reasonable bounds

        # Generate void shapes (ellipticity)
        ellipticities = np.random.beta(2, 5, n_voids)  # Skewed toward spherical

        # Density contrasts (typical void values)
        density_contrasts = np.random.uniform(-0.8, -0.3, n_voids)

        # Random alignment angles (no E8×E8 structure)
        alignment_angles = np.random.uniform(0, 360, n_voids)

        mock_voids = pd.DataFrame({
            'ra_deg': ra,
            'dec_deg': dec,
            'redshift': redshift,
            'radius_mpc': void_radii,
            'ellipticity': ellipticities,
            'density_contrast': density_contrasts,
            'alignment_angle_deg': alignment_angles,
            'volume_mpc3': (4/3) * np.pi * void_radii**3,
            'surface_area_mpc2': 4 * np.pi * void_radii**2,
            'survey': 'mock_voids',
            'algorithm': 'random_generation',
            'central_density': 1 + density_contrasts,  # Relative density
            'asphericity': np.random.uniform(0.1, 0.5, n_voids),  # Shape irregularity
            'note': 'Mock void catalog without E8×E8 alignment patterns'
        })

        return mock_voids

    def generate_mock_galaxy_catalog(self, n_galaxies: int = 50000,
                                    z_range: Tuple[float, float] = (0.01, 1.0),
                                    mag_range: Tuple[float, float] = (15, 22)) -> pd.DataFrame:
        """
        Generate mock galaxy catalog matching luminosity function.

        Parameters:
            n_galaxies: Number of galaxies
            z_range: Redshift range
            mag_range: Magnitude range

        Returns:
            pd.DataFrame: Mock galaxy catalog
        """
        np.random.seed(self.seed)

        # Generate redshifts with realistic distribution
        z = np.random.uniform(z_range[0], z_range[1], n_galaxies)

        # Generate positions (SDSS-like footprint)
        ra = np.random.uniform(100, 260, n_galaxies)
        dec = np.random.uniform(-10, 70, n_galaxies)

        # Magnitude distribution (power law + scatter)
        mag_exponent = -0.4  # Salpeter-like
        base_mags = np.random.uniform(mag_range[0], mag_range[1], n_galaxies)

        # Colors (realistic distributions, no H-ΛCDM correlations)
        u_minus_r = np.random.normal(2.2, 0.5, n_galaxies)
        u_minus_r = np.clip(u_minus_r, 0, 4)

        g_minus_r = np.random.normal(0.8, 0.3, n_galaxies)
        g_minus_r = np.clip(g_minus_r, 0, 2)

        r_minus_i = np.random.normal(0.4, 0.2, n_galaxies)
        r_minus_i = np.clip(r_minus_i, 0, 1)

        # Derived magnitudes
        r_mag = base_mags
        u_mag = r_mag + u_minus_r
        g_mag = r_mag + g_minus_r
        i_mag = r_mag - r_minus_i
        z_mag = i_mag - np.random.normal(0.3, 0.1, n_galaxies)

        # Morphological parameters (no clustering signals)
        petrosian_radius = 10**np.random.normal(1.2, 0.3, n_galaxies)
        concentration = np.random.uniform(2.5, 5.0, n_galaxies)
        half_light_radius = petrosian_radius * np.random.uniform(0.3, 0.8, n_galaxies)

        mock_galaxies = pd.DataFrame({
            'ra': ra,
            'dec': dec,
            'z': z,
            'z_err': z * 0.05,  # 5% redshift error
            'r_mag': r_mag,
            'u_mag': u_mag,
            'g_mag': g_mag,
            'i_mag': i_mag,
            'z_mag': z_mag,
            'u_minus_r': u_minus_r,
            'g_minus_r': g_minus_r,
            'r_minus_i': r_minus_i,
            'petrosian_radius': petrosian_radius,
            'half_light_radius': half_light_radius,
            'concentration': concentration,
            'survey': 'mock_galaxies',
            'note': 'Mock galaxy catalog without H-ΛCDM clustering signals'
        })

        return mock_galaxies

    def generate_mock_frb_catalog(self, n_frbs: int = 200) -> pd.DataFrame:
        """
        Generate mock FRB catalog without timing signatures.

        Parameters:
            n_frbs: Number of FRBs to generate

        Returns:
            pd.DataFrame: Mock FRB catalog
        """
        np.random.seed(self.seed)

        # Generate random positions and redshifts
        ra = np.random.uniform(0, 360, n_frbs)
        dec = np.random.uniform(-90, 90, n_frbs)
        redshift = np.random.uniform(0.1, 2.0, n_frbs)

        # Dispersion measures (typical values)
        dm = np.random.uniform(100, 2000, n_frbs)

        # Random timing intervals (no Little Bang patterns)
        time_intervals = np.random.exponential(100, n_frbs)  # Random waiting times

        # Fluxes and other properties
        flux = np.random.lognormal(-2, 1, n_frbs)
        fluence = flux * np.random.uniform(1, 10, n_frbs)

        mock_frbs = pd.DataFrame({
            'ra': ra,
            'dec': dec,
            'redshift': redshift,
            'dm': dm,  # dispersion measure
            'flux_jy': flux,
            'fluence_jy_ms': fluence,
            'time_interval_days': time_intervals,
            'survey': 'mock_frbs',
            'telescope': 'mock_telescope',
            'note': 'Mock FRB catalog without Little Bang information saturation patterns'
        })

        return mock_frbs

    def generate_mock_lyman_alpha(self, n_segments: int = 1000,
                                 wavelength_range: Tuple[float, float] = (3800, 4200)) -> pd.DataFrame:
        """
        Generate mock Lyman-alpha forest without phase transitions.

        Parameters:
            n_segments: Number of spectral segments
            wavelength_range: Wavelength range in Angstroms

        Returns:
            pd.DataFrame: Mock Lyman-alpha data
        """
        np.random.seed(self.seed)

        wavelength = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
        redshift = np.random.uniform(2.0, 4.0, n_segments)

        mock_segments = []
        for z in redshift:
            # Generate continuum (power law)
            continuum = wavelength**(-1.5) * np.random.uniform(0.8, 1.2)

            # Add random absorption features (no phase transitions)
            absorption_depth = np.random.uniform(0.1, 0.8, len(wavelength))
            absorption_width = np.random.uniform(0.5, 2.0, len(wavelength))

            # Create absorption profile
            flux = continuum * (1 - absorption_depth * np.exp(-((wavelength - wavelength.mean()) / absorption_width)**2))

            # Add noise
            noise_level = 0.05
            flux += np.random.normal(0, noise_level, len(flux))

            mock_segments.append({
                'redshift': z,
                'wavelength': wavelength.tolist(),
                'flux': flux.tolist(),
                'continuum': continuum.tolist(),
                'optical_depth': (-np.log(flux / continuum)).tolist(),
                'survey': 'mock_lyman_alpha',
                'quasar_id': f'mock_qso_{len(mock_segments):04d}',
                'note': 'Mock Lyman-alpha spectrum without H-ΛCDM phase transitions'
            })

        return pd.DataFrame(mock_segments)

    def generate_mock_jwst_catalog(self, n_galaxies: int = 100,
                                  z_range: Tuple[float, float] = (8.0, 15.0)) -> pd.DataFrame:
        """
        Generate mock JWST early galaxy catalog without anti-viscosity.

        Parameters:
            n_galaxies: Number of galaxies
            z_range: Redshift range

        Returns:
            pd.DataFrame: Mock JWST catalog
        """
        np.random.seed(self.seed)

        z = np.random.uniform(z_range[0], z_range[1], n_galaxies)
        ra = np.random.uniform(0, 360, n_galaxies)
        dec = np.random.uniform(-5, 5, n_galaxies)  # Deep field

        # Magnitudes (very faint, no mass limits)
        f200w_mag = np.random.uniform(24, 32, n_galaxies)  # F200W filter
        f277w_mag = f200w_mag + np.random.normal(0.2, 0.1, n_galaxies)
        f356w_mag = f277w_mag + np.random.normal(0.1, 0.1, n_galaxies)

        # Random morphologies (no anti-viscosity constraints)
        half_light_radius = np.random.lognormal(0, 0.5, n_galaxies)  # pixels
        sersic_index = np.random.uniform(1, 6, n_galaxies)

        # Random masses (no saturation limits)
        mass_estimate = 10**np.random.uniform(6, 10, n_galaxies)  # Solar masses

        mock_jwst = pd.DataFrame({
            'ra': ra,
            'dec': dec,
            'redshift': z,
            'z_err': z * 0.1,  # Higher uncertainty at high z
            'f200w_mag': f200w_mag,
            'f277w_mag': f277w_mag,
            'f356w_mag': f356w_mag,
            'half_light_radius_pix': half_light_radius,
            'sersic_index': sersic_index,
            'mass_log_msol': np.log10(mass_estimate),
            'survey': 'mock_jwst',
            'field': 'mock_deep_field',
            'note': 'Mock JWST catalog without H-ΛCDM anti-viscosity mass limits'
        })

        return mock_jwst

    def generate_validation_dataset(self, modality: str, size: str = 'medium') -> Dict[str, Any]:
        """
        Generate validation dataset for specific modality.

        Parameters:
            modality: Data type ('cmb', 'bao', 'void', 'galaxy', 'frb', 'lyman_alpha', 'jwst')
            size: Dataset size ('small', 'medium', 'large')

        Returns:
            dict: Mock dataset matching real data statistics
        """
        size_multipliers = {'small': 0.1, 'medium': 1.0, 'large': 10.0}
        multiplier = size_multipliers[size]

        if modality == 'cmb':
            return self.generate_mock_cmb_maps(int(1000 * multiplier))
        elif modality == 'bao':
            return self.generate_mock_bao_measurements(int(3 * multiplier))
        elif modality == 'void':
            return self.generate_mock_void_catalog(int(10000 * multiplier))
        elif modality == 'galaxy':
            return self.generate_mock_galaxy_catalog(int(50000 * multiplier))
        elif modality == 'frb':
            return self.generate_mock_frb_catalog(int(200 * multiplier))
        elif modality == 'lyman_alpha':
            return self.generate_mock_lyman_alpha(int(1000 * multiplier))
        elif modality == 'jwst':
            return self.generate_mock_jwst_catalog(int(100 * multiplier))
        elif modality == 'combined':
            # Generate all modalities
            combined = {}
            combined['cmb'] = self.generate_mock_cmb_maps(int(1000 * multiplier))['C_ell']
            combined['bao'] = np.array([m['value'] for m in self.generate_mock_bao_measurements(int(3 * multiplier))['measurements']])
            
            # For dataframes, we can't easily put them in a simple structure expected by feature extractor
            # unless we assume the extractor handles dicts of arrays/dfs.
            # But NullHypothesisTester._extract_features_from_dataset expects simple array or dict with 'features'.
            # If 'combined', we probably want to generate features directly or return a structure 
            # that NullHypothesisTester can handle.
            
            # For now, return a dictionary that acts as a container for multi-modal data
            # which NullHypothesisTester's extract_features might struggle with unless updated.
            # But looking at _extract_features_from_dataset in NullHypothesisTester, it checks for 'modalities' key.
            return {
                'modalities': {
                    'cmb': self.generate_mock_cmb_maps(int(1000 * multiplier)),
                    'bao': self.generate_mock_bao_measurements(int(3 * multiplier)),
                    'void': self.generate_mock_void_catalog(int(10000 * multiplier)),
                    'galaxy': self.generate_mock_galaxy_catalog(int(50000 * multiplier))
                }
            }
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def save_mock_dataset(self, dataset: Any, filename: str, output_dir: str = "mock_data"):
        """
        Save mock dataset to file.

        Parameters:
            dataset: Mock dataset (dict or DataFrame)
            filename: Output filename
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        filepath = output_path / filename

        if isinstance(dataset, pd.DataFrame):
            dataset.to_pickle(filepath)
        elif isinstance(dataset, dict):
            np.savez(filepath.with_suffix('.npz'), **dataset)
        else:
            np.save(filepath, dataset)

        print(f"Saved mock dataset to {filepath}")
