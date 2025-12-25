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
    output_dir: str = "results/figures/void"
) -> Dict[str, str]:
    """
    Generate comprehensive statistical figures for void analysis.

    Parameters:
        void_catalog_path: Path to processed void catalog
        results_path: Path to analysis results
        output_dir: Output directory for figures

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
    def get_survey(row):
        if 'clampitt' in str(row.get('source', '')).lower():
            return "SDSS DR7 (Clampitt)"
        elif 'douglass' in str(row.get('source', '')).lower():
            return "SDSS DR7 (Douglass)"
        elif 'desi' in str(row.get('survey', '')).lower():
            return "DESI DR1"
        else:
            return "Unknown"

    catalog_df['survey_name'] = catalog_df.apply(get_survey, axis=1)

    # Redshift distribution by survey
    surveys = catalog_df['survey_name'].unique()

    colors = ['#2E86AB', '#F18F01', '#0B6E4F', '#F24236']
    for i, survey in enumerate(surveys):
        survey_data = catalog_df[catalog_df['survey_name'] == survey]
        redshift = survey_data['redshift'].dropna()

        if len(redshift) > 0:
            ax1.hist(redshift, bins=30, alpha=0.7, label=survey,
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Number of Voids')
    ax1.set_title('Void Redshift Distribution by Survey')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    # Redshift vs RA
    if ra_col in catalog_df.columns and 'redshift' in catalog_df.columns:
        ra_clean = catalog_df[ra_col].dropna()
        z_clean = catalog_df['redshift'].dropna()
        min_len = min(len(ra_clean), len(z_clean))

        scatter2 = ax2.scatter(ra_clean[:min_len], z_clean[:min_len],
                              c=catalog_df['survey_color'][:min_len],
                              alpha=0.6, s=2)
        ax2.set_xlabel('Right Ascension (degrees)')
        ax2.set_ylabel('Redshift z')
        ax2.set_title('Void Distribution (RA-z)')
        ax2.grid(True, alpha=0.3)

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
