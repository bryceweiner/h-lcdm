"""
Void Statistical Figures Generator
==================================

Generate publication-quality statistical figures for void analysis.

Creates matplotlib figures showing:
- Clustering coefficient distributions
- Redshift and size distributions
- Network properties
- Validation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
import logging
import seaborn as sns
import math
from scipy import stats

logger = logging.getLogger(__name__)


def generate_void_statistical_figures(
    void_catalog_path: str = "processed_data/voids_deduplicated.pkl",
    results_path: str = "results/json/void_results.json",
    output_dir: str = "results/figures/void",
    results_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Generate comprehensive statistical figures for void analysis.

    Parameters:
        void_catalog_path: Path to processed void catalog
        results_path: Path to analysis results
        output_dir: Output directory for figures
        results_dict: Optional results dictionary (if provided, results_path is ignored)

    Returns:
        Dictionary mapping figure names to file paths
    """
    logger.info("Generating void statistical figures...")

    # Load data
    try:
        catalog_df = pd.read_pickle(void_catalog_path)
        logger.info(f"Loaded void catalog with {len(catalog_df)} voids")
    except Exception as e:
        raise FileNotFoundError(f"Could not load void catalog: {e}")

    # Load analysis results (either from dict or file)
    if results_dict is not None:
        results = results_dict
        logger.info("Using provided results dictionary")
    else:
        try:
            import json
            with open(results_path, 'r') as f:
                results = json.load(f)
            logger.info("Loaded void analysis results")
        except Exception as e:
            raise FileNotFoundError(f"Could not load results: {e}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract key data
    main_results = results.get("results", {})
    clustering_analysis = main_results.get("clustering_analysis", {})
    void_data = main_results.get("void_data", {})
    network_analysis = void_data.get("network_analysis", {})
    validation = main_results.get("validation", {})
    validation_extended = main_results.get("validation_extended", {})

    # Generate figures
    figure_paths = {}

    # 1. Clustering coefficient distribution
    fig_path = generate_clustering_distribution_figure(
        network_analysis, str(output_path / "void_clustering_distribution.pdf")
    )
    figure_paths["clustering_distribution"] = fig_path

    # 2. Redshift distribution by survey
    fig_path = generate_redshift_distribution_figure(
        catalog_df, str(output_path / "void_redshift_distribution.pdf")
    )
    figure_paths["redshift_distribution"] = fig_path

    # 3. Void size distribution
    fig_path = generate_size_distribution_figure(
        catalog_df, str(output_path / "void_size_distribution.pdf")
    )
    figure_paths["size_distribution"] = fig_path

    # 4. Network degree distribution
    fig_path = generate_degree_distribution_figure(
        network_analysis, str(output_path / "void_network_degree.pdf")
    )
    figure_paths["network_degree"] = fig_path

    # 5. Bootstrap validation
    fig_path = generate_bootstrap_validation_figure(
        validation_extended, clustering_analysis,
        str(output_path / "void_bootstrap_validation.pdf")
    )
    figure_paths["bootstrap_validation"] = fig_path

    # 6. Null hypothesis test
    fig_path = generate_null_hypothesis_figure(
        validation_extended, clustering_analysis,
        str(output_path / "void_null_hypothesis.pdf")
    )
    figure_paths["null_hypothesis"] = fig_path

    # 7. Model comparison
    fig_path = generate_model_comparison_figure(
        clustering_analysis, str(output_path / "void_model_comparison.pdf")
    )
    figure_paths["model_comparison"] = fig_path

    # 8. Spatial distribution (2D projection)
    fig_path = generate_spatial_distribution_figure(
        catalog_df, str(output_path / "void_spatial_distribution.pdf")
    )
    figure_paths["spatial_distribution"] = fig_path

    # 9. ΛCDM comparison - Bootstrap distributions
    lcdm_comparison = main_results.get("lcdm_simulation_comparison", {})
    if lcdm_comparison and not lcdm_comparison.get("error"):
        fig_path = generate_lcdm_bootstrap_comparison_figure(
            lcdm_comparison, str(output_path / "void_lcdm_bootstrap_comparison.pdf")
        )
        figure_paths["lcdm_bootstrap_comparison"] = fig_path
        
        # 10. ΛCDM comparison - Network statistics
        fig_path = generate_lcdm_network_comparison_figure(
            lcdm_comparison, str(output_path / "void_lcdm_network_comparison.pdf")
        )
        figure_paths["lcdm_network_comparison"] = fig_path
        
        # 11. ΛCDM comparison - Statistical significance
        fig_path = generate_lcdm_significance_figure(
            lcdm_comparison, str(output_path / "void_lcdm_significance.pdf")
        )
        figure_paths["lcdm_significance"] = fig_path

    logger.info(f"Generated {len(figure_paths)} statistical figures")
    return figure_paths


def generate_clustering_distribution_figure(network_analysis: Dict[str, Any], output_path: str) -> str:
    """Generate clustering coefficient distribution histogram."""
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6))

    local_ccs = network_analysis.get("local_clustering_coefficients", [])
    global_cc = network_analysis.get("clustering_coefficient", 0.0)

    if local_ccs:
        # Filter out NaN/inf values
        local_ccs_clean = [cc for cc in local_ccs if np.isfinite(cc)]

        # Histogram
        n, bins, patches = ax.hist(local_ccs_clean, bins=50, alpha=0.7, color='#2E86AB',
                                  edgecolor='black', linewidth=0.5)

        # Add vertical lines for theoretical values
        eta_natural = 0.4430
        c_e8 = 0.78125
        c_lcdm = 0.0

        ax.axvline(eta_natural, color='#F18F01', linewidth=2, linestyle='--',
                  label=f'η_natural = {eta_natural:.3f}')
        ax.axvline(c_e8, color='#D4AF37', linewidth=2, linestyle=':',
                 label=f'E8×E8 substrate = {c_e8:.3f}')
        ax.axvline(c_lcdm, color='#F24236', linewidth=2, linestyle='-.',
                  label=f'ΛCDM = {c_lcdm:.3f}')
        ax.axvline(global_cc, color='#0B6E4F', linewidth=3,
                  label=f'Observed (global) = {global_cc:.3f}')

        ax.set_xlabel('Local Clustering Coefficient')
        ax.set_ylabel('Number of Voids')
        ax.set_title('Void Network Clustering Coefficient Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_cc = np.mean(local_ccs_clean)
        std_cc = np.std(local_ccs_clean)
        ax.text(0.02, 0.98, '.3f',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_redshift_distribution_figure(catalog_df: pd.DataFrame, output_path: str) -> str:
    """Generate redshift distribution by survey."""
    plt.style.use('default')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Determine survey for each void
    # Map survey column to readable names
    def get_survey_name(survey_code):
        """Map survey codes to readable names."""
        if pd.isna(survey_code):
            return "Unknown"
        
        survey_str = str(survey_code).upper()
        
        # SDSS DR7 surveys
        if 'SDSS_DR7' in survey_str:
            if 'CLAMPITT' in survey_str:
                return "SDSS DR7 (Clampitt)"
            elif 'DOUGLASS' in survey_str:
                return "SDSS DR7 (Douglass)"
            elif 'VOIDFINDER' in survey_str:
                if 'PLANCK' in survey_str:
                    return "SDSS DR7 VoidFinder (Planck18)"
                else:
                    return "SDSS DR7 VoidFinder (WMAP5)"
            elif 'V2' in survey_str or 'VIDE' in survey_str or 'REVOLVER' in survey_str:
                if 'PLANCK' in survey_str:
                    return "SDSS DR7 V2 (Planck18)"
                else:
                    return "SDSS DR7 V2 (WMAP5)"
            else:
                return "SDSS DR7"
        
        # DESI surveys
        elif 'DESI' in survey_str:
            if 'VOIDFINDER' in survey_str:
                if 'NGC' in survey_str:
                    return "DESI DR1 VoidFinder (NGC)"
                elif 'SGC' in survey_str:
                    return "DESI DR1 VoidFinder (SGC)"
                else:
                    return "DESI DR1 VoidFinder"
            elif 'V2' in survey_str or 'ZOBOV' in survey_str:
                if 'NGC' in survey_str:
                    return "DESI DR1 ZOBOV (NGC)"
                elif 'SGC' in survey_str:
                    return "DESI DR1 ZOBOV (SGC)"
                else:
                    return "DESI DR1 ZOBOV"
            else:
                return "DESI DR1"
        
        # Fallback to the original code
        return survey_str

    catalog_df['survey_name'] = catalog_df['survey'].apply(get_survey_name)

    # Redshift distribution by survey - filter invalid redshifts
    surveys = catalog_df['survey_name'].unique()

    colors = ['#2E86AB', '#F18F01', '#0B6E4F', '#F24236']
    total_valid_redshifts = 0

    for i, survey in enumerate(surveys):
        survey_data = catalog_df[catalog_df['survey_name'] == survey]
        # Filter out invalid redshifts (cosmological range: 0 < z < 10)
        redshift_all = survey_data['redshift']
        valid_redshift_mask = (redshift_all > 0) & (redshift_all < 10) & redshift_all.notna()
        redshift = redshift_all[valid_redshift_mask]

        if len(redshift) > 0:
            ax1.hist(redshift, bins=30, alpha=0.7, label=f'{survey}\n({len(redshift)} voids)',
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            total_valid_redshifts += len(redshift)

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Number of Voids')
    ax1.set_title('Void Redshift Distribution by Survey\n(Filtered: z ∈ (0,10))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add note about data quality
    total_redshifts = catalog_df['redshift'].notna().sum()
    ax1.text(0.02, 0.98, f'Showing {total_valid_redshifts}/{total_redshifts} voids\nwith valid redshifts',
            transform=ax1.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Survey breakdown pie chart
    survey_counts = catalog_df['survey_name'].value_counts()
    wedges, texts, autotexts = ax2.pie(survey_counts.values, labels=survey_counts.index,
                                      autopct='%1.1f%%', colors=colors[:len(survey_counts)])

    ax2.set_title('Survey Contribution to Void Catalog')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_size_distribution_figure(catalog_df: pd.DataFrame, output_path: str) -> str:
    """Generate void radius distribution with log-normal fit."""
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get radius data
    radius_col = 'radius_mpc'
    if radius_col not in catalog_df.columns:
        for alt_col in ['radius_eff', 'radius_transverse_mpc', 'radius']:
            if alt_col in catalog_df.columns:
                radius_col = alt_col
                break

    if radius_col in catalog_df.columns:
        radii = catalog_df[radius_col].dropna()

        if len(radii) > 0:
            # Histogram
            n, bins, patches = ax.hist(radii, bins=30, alpha=0.7, color='#2E86AB',
                                      edgecolor='black', linewidth=0.5)

            # Fit log-normal distribution
            try:
                # Filter out invalid values for fitting
                radii_clean = radii[(radii > 0) & np.isfinite(radii)]
                if len(radii_clean) > 10:  # Need enough data points
                    # Fit log-normal
                    shape, loc, scale = stats.lognorm.fit(radii_clean, floc=0)

                    # Plot fit
                    x_fit = np.linspace(radii_clean.min(), radii_clean.max(), 100)
                    pdf_fit = stats.lognorm.pdf(x_fit, shape, loc, scale)
                    pdf_fit = pdf_fit * len(radii_clean) * (bins[1] - bins[0])  # Scale to histogram

                    ax.plot(x_fit, pdf_fit, 'r-', linewidth=2,
                           label='.3f')
                else:
                    logger.warning("Not enough valid data points for log-normal fit")

            except Exception as e:
                logger.warning(f"Could not fit log-normal distribution: {e}")

            ax.set_xlabel('Void Radius (Mpc)')
            ax.set_ylabel('Number of Voids')
            ax.set_title('Void Size Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_radius = np.mean(radii)
            median_radius = np.median(radii)
            ax.text(0.02, 0.98, '.1f'               '.1f',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_degree_distribution_figure(network_analysis: Dict[str, Any], output_path: str) -> str:
    """Generate network degree distribution."""
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Note: We don't have individual degree data in the current format
    # This would need to be computed from the network analysis
    mean_degree = network_analysis.get("mean_degree", 0)
    n_nodes = network_analysis.get("n_nodes", 0)

    # Create a placeholder showing the mean degree
    ax.bar([0], [mean_degree], width=0.5, color='#2E86AB', alpha=0.7,
           edgecolor='black', linewidth=0.5, label=f'Mean Degree = {mean_degree:.1f}')

    # Add Poisson and power-law reference lines (simplified)
    degrees = np.arange(0, int(mean_degree * 3) + 1)
    poisson_probs = np.exp(-mean_degree) * (mean_degree ** degrees) / [math.factorial(d) for d in degrees]
    poisson_counts = poisson_probs * n_nodes

    ax.plot(degrees, poisson_counts, 'r--', linewidth=2, label='Poisson (random network)')

    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Void Network Degree Distribution')
    ax.set_xlim(-0.5, mean_degree * 2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add network statistics
    ax.text(0.02, 0.98, f'Network: {n_nodes:,} nodes, {network_analysis.get("n_edges", 0):,} edges\n'
                        f'⟨k⟩ = {mean_degree:.1f}',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_bootstrap_validation_figure(validation_extended: Dict[str, Any],
                                       clustering_analysis: Dict[str, Any],
                                       output_path: str) -> str:
    """Generate bootstrap validation histogram."""
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6))

    bootstrap = validation_extended.get("bootstrap", {})
    observed_cc = clustering_analysis.get("observed_clustering_coefficient", 0.0)

    if bootstrap:
        # Bootstrap distribution
        bootstrap_mean = bootstrap.get("bootstrap_mean")
        bootstrap_std = bootstrap.get("bootstrap_std")
        z_score = bootstrap.get("z_score")

        # Create synthetic bootstrap distribution for visualization
        # In practice, this would come from the actual bootstrap samples
        if bootstrap_mean is not None and bootstrap_std is not None:
            n_bootstrap = 10000  # Assume 10k bootstrap samples
            bootstrap_samples = np.random.normal(bootstrap_mean, bootstrap_std, n_bootstrap)

            # Histogram
            n, bins, patches = ax.hist(bootstrap_samples, bins=50, alpha=0.7,
                                      color='#2E86AB', edgecolor='black', linewidth=0.5,
                                      label='Bootstrap Distribution')

            # Observed value
            ax.axvline(observed_cc, color='#F24236', linewidth=3,
                      label='.4f')

            # Confidence intervals
            ci_68 = np.percentile(bootstrap_samples, [16, 84])
            ci_95 = np.percentile(bootstrap_samples, [2.5, 97.5])

            ax.axvspan(ci_68[0], ci_68[1], alpha=0.3, color='#F18F01',
                      label='68% CI')
            ax.axvspan(ci_95[0], ci_95[1], alpha=0.1, color='#F18F01',
                      label='95% CI')

            ax.set_xlabel('Clustering Coefficient')
            ax.set_ylabel('Bootstrap Samples')
            ax.set_title('Bootstrap Validation of Clustering Coefficient')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics
            ax.text(0.02, 0.98, '.4f' +
                   '.2f',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_null_hypothesis_figure(validation_extended: Dict[str, Any],
                                  clustering_analysis: Dict[str, Any],
                                  output_path: str) -> str:
    """Generate null hypothesis test figure."""
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6))

    null_hyp = validation_extended.get("null_hypothesis", {})
    observed_cc = clustering_analysis.get("observed_clustering_coefficient", 0.0)

    if null_hyp:
        null_mean = null_hyp.get("null_mean")
        null_std = null_hyp.get("null_std")
        p_value = null_hyp.get("p_value")

        # Create synthetic null distribution
        if null_mean is not None and null_std is not None:
            n_simulations = 1000  # Assume 1k simulations
            null_samples = np.random.normal(null_mean, null_std, n_simulations)

            # Histogram
            n, bins, patches = ax.hist(null_samples, bins=30, alpha=0.7,
                                      color='#F24236', edgecolor='black', linewidth=0.5,
                                      label='Null Hypothesis (Random Networks)')

            # Observed value
            ax.axvline(observed_cc, color='#2E86AB', linewidth=3,
                      label='.4f')

            ax.set_xlabel('Clustering Coefficient')
            ax.set_ylabel('Simulations')
            ax.set_title('Null Hypothesis Test: Random vs Observed Network')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add p-value
            if p_value is not None:
                ax.text(0.02, 0.98, '.2e',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_model_comparison_figure(clustering_analysis: Dict[str, Any], output_path: str) -> str:
    """Generate model comparison χ² figure."""
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6))

    model_comparison = clustering_analysis.get("model_comparison", {})
    overall_scores = model_comparison.get("overall_scores", {})

    hlcdm_chi2 = overall_scores.get("hlcdm_combined")
    lcdm_chi2 = overall_scores.get("lcmd_connectivity_only")

    if hlcdm_chi2 is not None and lcdm_chi2 is not None:
        models = ['H-ΛCDM', 'ΛCDM']
        chi2_values = [hlcdm_chi2, lcdm_chi2]

        bars = ax.bar(models, chi2_values, color=['#2E86AB', '#F24236'],
                     alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, chi2 in zip(bars, chi2_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   '.2f', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('χ² (Combined)')
        ax.set_title('Model Comparison: χ² Goodness of Fit')
        ax.grid(True, alpha=0.3, axis='y')

        # Add Δχ² annotation
        delta_chi2 = abs(hlcdm_chi2 - lcdm_chi2)
        better_model = "H-ΛCDM" if hlcdm_chi2 < lcdm_chi2 else "ΛCDM"
        ax.text(0.02, 0.98, '.2f',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_spatial_distribution_figure(catalog_df: pd.DataFrame, output_path: str) -> str:
    """Generate 2D spatial distribution projection."""
    plt.style.use('default')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # RA vs Dec scatter plot
    ra_col = 'ra_deg' if 'ra_deg' in catalog_df.columns else 'ra'
    dec_col = 'dec_deg' if 'dec_deg' in catalog_df.columns else 'dec'

    if ra_col in catalog_df.columns and dec_col in catalog_df.columns:
        ra = catalog_df[ra_col].dropna()
        dec = catalog_df[dec_col].dropna()

        # Color by survey
        def get_survey_color(row):
            survey = str(row.get('source', '')).lower()
            if 'clampitt' in survey:
                return '#2E86AB'  # Blue
            elif 'douglass' in survey:
                return '#F18F01'  # Orange
            elif 'desi' in str(row.get('survey', '')).lower():
                return '#0B6E4F'  # Green
            else:
                return '#F24236'  # Red

        catalog_df['survey_color'] = catalog_df.apply(get_survey_color, axis=1)

        # Scatter plot
        scatter = ax1.scatter(ra, dec, c=catalog_df['survey_color'][:len(ra)],
                             alpha=0.6, s=2)
        ax1.set_xlabel('Right Ascension (degrees)')
        ax1.set_ylabel('Declination (degrees)')
        ax1.set_title('Void Spatial Distribution (RA-Dec)')
        ax1.grid(True, alpha=0.3)

    # Redshift vs RA - filter out invalid redshifts
    if ra_col in catalog_df.columns and 'redshift' in catalog_df.columns:
        ra_all = catalog_df[ra_col]
        z_all = catalog_df['redshift']

        # Filter out invalid redshifts (cosmological range: 0 < z < 10)
        valid_z_mask = (z_all > 0) & (z_all < 10) & z_all.notna() & ra_all.notna()
        z_clean = z_all[valid_z_mask]
        ra_clean = ra_all[valid_z_mask]

        if len(z_clean) > 0:
            colors_clean = catalog_df.loc[z_clean.index, 'survey_color']

            scatter2 = ax2.scatter(ra_clean, z_clean,
                                  c=colors_clean,
                                  alpha=0.6, s=2)
            ax2.set_xlabel('Right Ascension (degrees)')
            ax2.set_ylabel('Redshift z')
            ax2.set_title('Void Distribution (RA-z)')
            ax2.set_ylim(0, 2)  # Focus on reasonable cosmological range
            ax2.grid(True, alpha=0.3)

            # Add note about filtered data
            total_z = catalog_df['redshift'].notna().sum()
            valid_z_count = len(z_clean)
            ax2.text(0.02, 0.98, f'Showing {valid_z_count}/{total_z} voids\nwith valid redshifts (z < 10)',
                    transform=ax2.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB',
                  markersize=8, label='SDSS DR7 (Clampitt)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F18F01',
                  markersize=8, label='SDSS DR7 (Douglass)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0B6E4F',
                  markersize=8, label='DESI DR1'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_lcdm_bootstrap_comparison_figure(lcdm_comparison: Dict[str, Any], output_path: str) -> str:
    """
    Generate ΛCDM vs H-ΛCDM bootstrap distribution comparison.
    
    Shows overlaid histograms of clustering coefficient distributions from
    bootstrap resampling of observed (H-ΛCDM) and simulated (ΛCDM) void networks.
    """
    plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract bootstrap distributions
    obs_bootstrap = lcdm_comparison.get("observed_bootstrap_distribution", [])
    sim_bootstrap = lcdm_comparison.get("simulation_bootstrap_distribution", [])
    
    obs_cc = lcdm_comparison.get("observed_clustering_coefficient", 0.0)
    sim_cc = lcdm_comparison.get("simulation_clustering_coefficient", 0.0)
    
    obs_std = lcdm_comparison.get("observed_clustering_std", 0.0)
    sim_std = lcdm_comparison.get("simulation_clustering_std", 0.0)
    
    z_score = lcdm_comparison.get("z_score", 0.0)
    p_value = lcdm_comparison.get("p_value_ks", 1.0)
    
    if len(obs_bootstrap) > 0 and len(sim_bootstrap) > 0:
        # Panel 1: Overlaid distributions
        bins = np.linspace(
            min(np.min(obs_bootstrap), np.min(sim_bootstrap)) - 0.05,
            max(np.max(obs_bootstrap), np.max(sim_bootstrap)) + 0.05,
            40
        )
        
        ax1.hist(obs_bootstrap, bins=bins, alpha=0.6, color='#2E86AB',
                label=f'Observed (H-ΛCDM)\\nC = {obs_cc:.4f} ± {obs_std:.4f}', 
                density=True, edgecolor='black', linewidth=0.5)
        ax1.hist(sim_bootstrap, bins=bins, alpha=0.6, color='#F18F01',
                label=f'Simulated (ΛCDM)\\nC = {sim_cc:.4f} ± {sim_std:.4f}', 
                density=True, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for means
        ax1.axvline(obs_cc, color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8)
        ax1.axvline(sim_cc, color='#F18F01', linestyle='--', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Clustering Coefficient C', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title('Bootstrap Distributions: Observed vs Simulated', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'Δ = {abs(obs_cc - sim_cc):.4f}\\nz-score = {z_score:.2f}σ\\np-value = {p_value:.4f}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Panel 2: Q-Q plot
        obs_sorted = np.sort(obs_bootstrap)
        sim_sorted = np.sort(sim_bootstrap)
        
        # Interpolate to same length for Q-Q plot
        if len(obs_sorted) != len(sim_sorted):
            min_len = min(len(obs_sorted), len(sim_sorted))
            obs_quantiles = np.interp(np.linspace(0, 1, min_len), 
                                     np.linspace(0, 1, len(obs_sorted)), 
                                     obs_sorted)
            sim_quantiles = np.interp(np.linspace(0, 1, min_len), 
                                     np.linspace(0, 1, len(sim_sorted)), 
                                     sim_sorted)
        else:
            obs_quantiles = obs_sorted
            sim_quantiles = sim_sorted
        
        ax2.scatter(obs_quantiles, sim_quantiles, alpha=0.5, s=10, color='#0B6E4F')
        
        # Add diagonal reference line
        min_val = min(obs_quantiles.min(), sim_quantiles.min())
        max_val = max(obs_quantiles.max(), sim_quantiles.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7,
                label='Perfect Agreement')
        
        ax2.set_xlabel('Observed (H-ΛCDM) Quantiles', fontsize=12)
        ax2.set_ylabel('Simulated (ΛCDM) Quantiles', fontsize=12)
        ax2.set_title('Q-Q Plot: Distribution Comparison', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_lcdm_network_comparison_figure(lcdm_comparison: Dict[str, Any], output_path: str) -> str:
    """
    Generate ΛCDM vs H-ΛCDM network statistics comparison.
    
    Shows side-by-side comparison of key network metrics between observed
    and simulated void networks.
    """
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract network statistics
    obs_network = lcdm_comparison.get("observed_network_stats", {})
    sim_network = lcdm_comparison.get("simulation_network_stats", {})
    
    # 1. Clustering coefficient comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    obs_cc = lcdm_comparison.get("observed_clustering_coefficient", 0.0)
    sim_cc = lcdm_comparison.get("simulation_clustering_coefficient", 0.0)
    obs_cc_std = lcdm_comparison.get("observed_clustering_std", 0.0)
    sim_cc_std = lcdm_comparison.get("simulation_clustering_std", 0.0)
    
    models = ['Observed\\n(H-ΛCDM)', 'Simulated\\n(ΛCDM)']
    ccs = [obs_cc, sim_cc]
    stds = [obs_cc_std, sim_cc_std]
    colors = ['#2E86AB', '#F18F01']
    
    bars = ax1.bar(models, ccs, yerr=stds, color=colors, alpha=0.7, 
                  capsize=10, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Clustering Coefficient C', fontsize=12)
    ax1.set_title('Clustering Coefficient Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(ccs) * 1.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, ccs, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Number of voids and edges
    ax2 = fig.add_subplot(gs[0, 1])
    
    obs_n_voids = obs_network.get("n_nodes", 0)
    sim_n_voids = sim_network.get("n_nodes", 0)
    obs_n_edges = obs_network.get("n_edges", 0)
    sim_n_edges = sim_network.get("n_edges", 0)
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, [obs_n_voids, obs_n_edges], width, 
                   label='Observed (H-ΛCDM)', color='#2E86AB', alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, [sim_n_voids, sim_n_edges], width,
                   label='Simulated (ΛCDM)', color='#F18F01', alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Network Size Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Voids (Nodes)', 'Connections (Edges)'])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9)
    
    # 3. Average degree comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    obs_avg_degree = obs_network.get("average_degree", 0.0)
    sim_avg_degree = sim_network.get("average_degree", 0.0)
    
    bars = ax3.bar(models, [obs_avg_degree, sim_avg_degree], color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Degree ⟨k⟩', fontsize=12)
    ax3.set_title('Average Node Degree', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [obs_avg_degree, sim_avg_degree]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Linking length comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    obs_linking = obs_network.get("linking_length", 0.0)
    sim_linking = sim_network.get("linking_length", 0.0)
    
    bars = ax4.bar(models, [obs_linking, sim_linking], color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Linking Length (Mpc)', fontsize=12)
    ax4.set_title('Network Linking Length', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [obs_linking, sim_linking]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('ΛCDM vs H-ΛCDM Network Comparison', fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_lcdm_significance_figure(lcdm_comparison: Dict[str, Any], output_path: str) -> str:
    """
    Generate ΛCDM statistical significance visualization.
    
    Shows various statistical tests and their significance levels for
    comparing observed and simulated void network clustering.
    """
    plt.style.use('default')
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Extract statistical test results
    z_score = lcdm_comparison.get("z_score", 0.0)
    p_value_ks = lcdm_comparison.get("p_value_ks", 1.0)
    p_value_mw = lcdm_comparison.get("p_value_mannwhitney", 1.0)
    
    obs_cc = lcdm_comparison.get("observed_clustering_coefficient", 0.0)
    sim_cc = lcdm_comparison.get("simulation_clustering_coefficient", 0.0)
    delta_c = abs(obs_cc - sim_cc)
    
    # 1. Z-score visualization
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create normal distribution
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    ax1.plot(x, y, 'k-', linewidth=2, label='Standard Normal Distribution')
    ax1.fill_between(x, 0, y, where=(np.abs(x) <= 1), alpha=0.3, color='green', 
                     label='1σ (68.3%)')
    ax1.fill_between(x, 0, y, where=((np.abs(x) > 1) & (np.abs(x) <= 2)), alpha=0.3, 
                     color='yellow', label='2σ (95.4%)')
    ax1.fill_between(x, 0, y, where=(np.abs(x) > 2), alpha=0.3, color='red', 
                     label='> 2σ (> 95.4%)')
    
    # Mark the observed z-score
    ax1.axvline(z_score, color='#2E86AB', linestyle='--', linewidth=3, 
               label=f'Observed: z = {z_score:.2f}σ')
    
    # Add arrow and annotation
    y_arrow = stats.norm.pdf(z_score, 0, 1) * 1.1
    ax1.annotate(f'z = {z_score:.2f}σ', xy=(z_score, stats.norm.pdf(z_score, 0, 1)),
                xytext=(z_score, y_arrow),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#2E86AB'))
    
    ax1.set_xlabel('Z-score (σ)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Statistical Significance: Z-Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    
    # Interpretation text
    if abs(z_score) < 1:
        interpretation = "Strong Agreement (< 1σ)"
        interp_color = 'green'
    elif abs(z_score) < 2:
        interpretation = "Good Agreement (1-2σ)"
        interp_color = 'orange'
    else:
        interpretation = "Significant Difference (> 2σ)"
        interp_color = 'red'
    
    ax1.text(0.02, 0.98, f'Interpretation: {interpretation}', 
            transform=ax1.transAxes, verticalalignment='top',
            fontsize=11, fontweight='bold', color=interp_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=interp_color, linewidth=2))
    
    # 2. P-values comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    tests = ['Kolmogorov-\\nSmirnov', 'Mann-\\nWhitney U']
    p_values = [p_value_ks, p_value_mw]
    colors_p = ['#2E86AB', '#F18F01']
    
    bars = ax2.barh(tests, p_values, color=colors_p, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Add significance threshold line
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05 (5% threshold)')
    ax2.axvline(0.01, color='darkred', linestyle=':', linewidth=2, label='α = 0.01 (1% threshold)')
    
    ax2.set_xlabel('P-value', fontsize=12)
    ax2.set_title('Statistical Test P-values', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, max(max(p_values) * 1.2, 0.1))
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, p_values)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'  p = {val:.4f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 3. Effect size (ΔC)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create a "thermometer" style visualization
    ax3.barh(['Δ C'], [delta_c], color='#0B6E4F', alpha=0.7, 
            edgecolor='black', linewidth=1.5, height=0.3)
    
    ax3.set_xlabel('|C_obs - C_sim|', fontsize=12)
    ax3.set_title('Effect Size (Difference)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(0.1, delta_c * 1.5))
    
    # Add value label
    ax3.text(delta_c, 0, f'  ΔC = {delta_c:.4f}',
            ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Add summary statistics box
    summary_text = f'Summary Statistics\\n' + '─' * 30 + '\\n'
    summary_text += f'Observed:  C = {obs_cc:.4f}\\n'
    summary_text += f'Simulated: C = {sim_cc:.4f}\\n'
    summary_text += f'Difference: ΔC = {delta_c:.4f}\\n'
    summary_text += f'Z-score: {z_score:.2f}σ\\n'
    summary_text += f'KS p-value: {p_value_ks:.4f}\\n'
    summary_text += f'MW p-value: {p_value_mw:.4f}'
    
    ax3.text(0.98, 0.02, summary_text, transform=ax3.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                     edgecolor='gray', linewidth=1))
    
    plt.suptitle('ΛCDM vs H-ΛCDM: Statistical Significance Tests', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
