"""
CMB Cold Spot Visualization
===========================

Generate publication-quality figures for Cold Spot QTEP analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def plot_cold_spot_temperature_map(cmb_map: np.ndarray,
                                   mask: np.ndarray,
                                   cold_spot_data: Dict[str, Any],
                                   output_path: Path,
                                   qtep_map: Optional[np.ndarray] = None) -> None:
    """
    Plot zoomed Cold Spot region with QTEP efficiency overlay at 50% opacity.
    
    Parameters:
        cmb_map: Full-sky CMB temperature map
        mask: Cold Spot region mask
        cold_spot_data: Extracted Cold Spot data with metadata
        output_path: Path to save figure
        qtep_map: Optional QTEP efficiency map to overlay at 50% opacity
    """
    if not HEALPY_AVAILABLE:
        logger.warning("healpy not available, skipping temperature map figure")
        return
    
    try:
        from PIL import Image
        import io
        
        # Get Cold Spot center from metadata
        l_center = cold_spot_data['metadata']['center_galactic_l']
        b_center = cold_spot_data['metadata']['center_galactic_b']
        
        if qtep_map is not None:
            # Create zoomed overlay with CMB + QTEP at 50% opacity
            # Render CMB temperature (base layer) - zoomed to Cold Spot
            fig1 = plt.figure(figsize=(10, 10))
            hp.gnomview(cmb_map,
                       rot=[l_center, b_center, 0],
                       reso=3.0,  # 3 arcmin resolution
                       xsize=1000,
                       title='',
                       cmap='RdBu_r',
                       min=-3*np.std(cmb_map),
                       max=3*np.std(cmb_map),
                       cbar=False,
                       notext=True)
            
            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
            buf1.seek(0)
            cmb_img = Image.open(buf1)
            plt.close(fig1)
            
            # Render QTEP efficiency (overlay layer) - same zoom
            fig2 = plt.figure(figsize=(10, 10))
            hp.gnomview(qtep_map,
                       rot=[l_center, b_center, 0],
                       reso=3.0,
                       xsize=1000,
                       title='',
                       cmap='viridis',
                       cbar=False,
                       notext=True)
            
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
            buf2.seek(0)
            qtep_img = Image.open(buf2)
            plt.close(fig2)
            
            # Convert to RGBA
            cmb_rgba = cmb_img.convert('RGBA')
            qtep_rgba = qtep_img.convert('RGBA')
            
            # Resize to match if needed
            if cmb_rgba.size != qtep_rgba.size:
                qtep_rgba = qtep_rgba.resize(cmb_rgba.size, Image.Resampling.LANCZOS)
            
            # Set QTEP to 25% opacity
            qtep_array = np.array(qtep_rgba, dtype=np.float32)
            qtep_array[:, :, 3] = 63.75  # 25% opacity
            qtep_rgba = Image.fromarray(qtep_array.astype(np.uint8), 'RGBA')
            
            # Composite: QTEP on top of CMB
            overlay = Image.alpha_composite(cmb_rgba, qtep_rgba)
            
            # Create final figure
            fig = plt.figure(figsize=(14, 12))
            ax = plt.subplot(1, 1, 1)
            ax.imshow(overlay)
            ax.axis('off')
            ax.set_title(f'Cold Spot Region (l={l_center}°, b={b_center}°)\n' +
                        'CMB Temperature + QTEP Efficiency Overlay (25% opacity)',
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add colorbars
            from matplotlib.colors import Normalize
            
            # CMB colorbar (base)
            ax_cbar_cmb = fig.add_axes([0.15, 0.08, 0.3, 0.02])
            norm_cmb = Normalize(vmin=-3*np.std(cmb_map), vmax=3*np.std(cmb_map))
            cb_cmb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_cmb, cmap='RdBu_r'),
                                 cax=ax_cbar_cmb, orientation='horizontal')
            cb_cmb.set_label('CMB Temperature [μK] (base layer)', fontsize=11)
            
            # QTEP colorbar (overlay)
            ax_cbar_qtep = fig.add_axes([0.55, 0.08, 0.3, 0.02])
            norm_qtep = Normalize(vmin=np.min(qtep_map), vmax=np.max(qtep_map))
            cb_qtep = plt.colorbar(plt.cm.ScalarMappable(norm=norm_qtep, cmap='viridis'),
                                  cax=ax_cbar_qtep, orientation='horizontal')
            cb_qtep.set_label('QTEP Efficiency η (25% overlay)', fontsize=11)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Cleanup
            buf1.close()
            buf2.close()
            
        else:
            # No QTEP map, just show CMB zoomed in
            hp.gnomview(cmb_map,
                       rot=[l_center, b_center, 0],
                       reso=3.0,
                       xsize=1000,
                       title=f'Cold Spot Region (l={l_center}°, b={b_center}°)',
                       unit='Temperature [μK]',
                       cmap='RdBu_r',
                       min=-3*np.std(cmb_map),
                       max=3*np.std(cmb_map))
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved Cold Spot temperature map to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create temperature map figure: {e}")
        import traceback
        logger.warning(traceback.format_exc())


def plot_angular_power_spectrum(ell: np.ndarray,
                                C_ell: np.ndarray,
                                test_results: Dict[str, Any],
                                output_path: Path) -> None:
    """
    Plot angular power spectrum C_ell vs ell with Gaussian comparison.
    
    Parameters:
        ell: Multipole array
        C_ell: Power spectrum values
        test_results: Test 2 results dictionary
        output_path: Path to save figure
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Only plot the physically meaningful range (l = 2 to 100)
        # High-l values have numerical artifacts from power spectrum conversion
        # Cut off before the systematic tail kicks in
        ell_max_plot = min(100, len(ell) - 1)
        ell_plot = ell[2:ell_max_plot]
        C_ell_plot = C_ell[2:ell_max_plot]
        
        # Filter out any non-positive values (numerical artifacts)
        valid_mask = C_ell_plot > 0
        ell_plot = ell_plot[valid_mask]
        C_ell_plot = C_ell_plot[valid_mask]
        
        if len(ell_plot) == 0:
            logger.warning("No valid power spectrum data to plot")
            return
        
        # Plot power spectrum
        ax.loglog(ell_plot, C_ell_plot, 'b-', linewidth=2, label='Observed CMB', alpha=0.8)
        
        # Add smooth power law fit for reference
        if len(ell_plot) > 10:
            # Fit in low-l range where Cold Spot signal would be
            ell_fit_range = ell_plot[ell_plot < 100]
            C_ell_fit_range = C_ell_plot[:len(ell_fit_range)]
            
            if len(ell_fit_range) > 5:
                log_ell_fit = np.log(ell_fit_range)
                log_C_fit = np.log(C_ell_fit_range)
                coeffs = np.polyfit(log_ell_fit, log_C_fit, 1)
                
                # Plot smooth fit
                C_ell_smooth = np.exp(coeffs[0] * np.log(ell_plot) + coeffs[1])
                ax.loglog(ell_plot, C_ell_smooth, 'g--', linewidth=2, alpha=0.6,
                         label='Smooth power law fit')
        
        # Highlight QTEP-predicted multipoles (DERIVED FROM η ≈ 2.257)
        # Base scale ℓ₀ = 18 (Cold Spot size ~10°)
        # QTEP predicts discrete transitions at: ℓ_n = ℓ₀ × η^n
        # Previous arbitrary [18, 36, 54, 72] were NOT QTEP-derived!
        eta = 2.257  # QTEP ratio
        ell_base = 18
        qtep_multipoles_derived = [
            ell_base,                    # ℓ₀ = 18
            int(ell_base * eta),         # ℓ₁ ≈ 41
            int(ell_base * eta**2),      # ℓ₂ ≈ 92
        ]
        qtep_multipoles_derived = [ell for ell in qtep_multipoles_derived if 2 < ell <= ell_max_plot]
        
        for i, qtep_ell in enumerate(qtep_multipoles_derived):
            label = f'QTEP ℓ_{i}={qtep_ell} (η^{i})' if i < 2 else f'QTEP ℓ_{i}={qtep_ell}'
            ax.axvline(qtep_ell, color='red', linestyle='--', alpha=0.6, linewidth=2.0, label=label)
            
            # Add annotation for regime interpretation
            if i == 0:
                regime_name = 'SW→Acoustic'
            elif i == 1:
                regime_name = 'Early Acoustic'
            elif i == 2:
                regime_name = 'Peak→Damping'
            else:
                regime_name = f'Transition {i}'
            
            # Annotate regime boundary
            y_pos = ax.get_ylim()[1] * 0.5 / (i + 1)
            ax.text(qtep_ell, y_pos, regime_name, rotation=90, 
                   fontsize=8, alpha=0.7, color='red',
                   verticalalignment='bottom', horizontalalignment='right')
        
        # Use regular 'l' instead of script small l to avoid font issues
        ax.set_xlabel('Multipole l', fontsize=12)
        ax.set_ylabel('Power Spectrum C_l [μK²]', fontsize=12)
        ax.set_title('Cold Spot Angular Power Spectrum (l = 2-100)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        # Set reasonable axis limits
        ax.set_xlim(2, ell_max_plot)
        
        # Add annotation with test results
        discrete_score = test_results.get('discrete_feature_score', 0)
        gaussian_p = test_results.get('gaussian_p_value', 1.0)
        result = test_results.get('result', 'UNKNOWN')
        
        textstr = f'Discrete feature score: {discrete_score:.2f}σ\n'
        textstr += f'Gaussian p-value: {gaussian_p:.3f}\n'
        textstr += f'Result: {result}'
        
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved angular power spectrum to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create power spectrum figure: {e}")
        import traceback
        logger.warning(traceback.format_exc())


def plot_qtep_efficiency_map(qtep_map: np.ndarray,
                            nside: int,
                            output_path: Path) -> None:
    """
    Plot QTEP efficiency map showing predicted spatial variations.
    
    Parameters:
        qtep_map: QTEP efficiency map
        nside: HEALPix resolution
        output_path: Path to save figure
    """
    if not HEALPY_AVAILABLE:
        logger.warning("healpy not available, skipping QTEP map figure")
        return
    
    try:
        # Create new figure - let healpy manage figure numbers to avoid conflicts
        plt.figure(figsize=(10, 6))
        
        hp.mollview(qtep_map,
                   title='QTEP Efficiency Map η(θ, φ)',
                   unit='Efficiency η',
                   cmap='viridis')
        
        # Don't use tight_layout with healpy plots
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved QTEP efficiency map to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create QTEP map figure: {e}")


def plot_spatial_correlation(cold_spot_map: np.ndarray,
                            qtep_map: np.ndarray,
                            mask: np.ndarray,
                            test_results: Dict[str, Any],
                            output_path: Path) -> None:
    """
    Plot cross-correlation visualization between Cold Spot and QTEP map.
    
    Parameters:
        cold_spot_map: Cold Spot temperature map
        qtep_map: QTEP efficiency map
        mask: Cold Spot region mask
        test_results: Test 3 results dictionary
        output_path: Path to save figure
    """
    try:
        # Extract valid pixels
        temp_valid = cold_spot_map[mask]
        efficiency_valid = qtep_map[mask]
        
        # Normalize for visualization
        temp_norm = (temp_valid - np.mean(temp_valid)) / (np.std(temp_valid) + 1e-10)
        efficiency_norm = (efficiency_valid - np.mean(efficiency_valid)) / (np.std(efficiency_valid) + 1e-10)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1.scatter(efficiency_norm, temp_norm, alpha=0.5, s=10)
        
        # Fit line
        if len(efficiency_norm) > 1:
            coeffs = np.polyfit(efficiency_norm, temp_norm, 1)
            x_fit = np.linspace(efficiency_norm.min(), efficiency_norm.max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label='Linear fit')
        
        correlation = test_results.get('correlation_coefficient', 0)
        ax1.set_xlabel('QTEP Efficiency (normalized)')
        ax1.set_ylabel('Temperature (normalized)')
        ax1.set_title(f'Spatial Correlation (r = {correlation:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2D density plot showing SPATIAL relationship (not just marginal distributions)
        # Note: Identical histograms ≠ spatial correlation!
        # This hexbin shows the actual pixel-by-pixel relationship density
        hexbin = ax2.hexbin(efficiency_norm, temp_norm, gridsize=20, cmap='Blues', 
                           mincnt=1, alpha=0.7)
        ax2.set_xlabel('QTEP Efficiency (normalized)')
        ax2.set_ylabel('Temperature (normalized)')
        ax2.set_title(f'Pixel-by-Pixel Relationship Density\n(r={correlation:.3f} measures this, not marginal distributions)')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add reference line for perfect correlation
        ax2.plot([-3, 3], [-3, 3], 'r--', alpha=0.3, linewidth=1, label='r=1 (perfect)')
        ax2.legend(fontsize=8)
        
        # Add colorbar for hexbin
        plt.colorbar(hexbin, ax=ax2, label='Pixel count')
        
        # Add test results annotation
        p_value = test_results.get('random_location_p_value', 1.0)
        result = test_results.get('result', 'UNKNOWN')
        
        textstr = f'Correlation: r = {correlation:.3f}\n'
        textstr += f'Random location p-value: {p_value:.3f}\n'
        textstr += f'Result: {result}'
        
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spatial correlation plot to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create correlation figure: {e}")


def plot_cmb_qtep_overlay(cmb_map: np.ndarray,
                         qtep_map: np.ndarray,
                         cold_spot_mask: np.ndarray,
                         cold_spot_center: Tuple[float, float],
                         nside: int,
                         output_path: Path) -> None:
    """
    Create true 50% opacity overlay: CMB temperature on top of QTEP efficiency.
    
    Parameters:
        cmb_map: CMB temperature map
        qtep_map: QTEP efficiency map
        cold_spot_mask: Mask for Cold Spot region
        cold_spot_center: (l, b) galactic coordinates of Cold Spot
        nside: HEALPix resolution
        output_path: Path to save figure
    """
    if not HEALPY_AVAILABLE:
        logger.warning("healpy not available, skipping overlay figure")
        return
    
    try:
        import matplotlib.colors as mcolors
        from matplotlib.colors import Normalize
        from PIL import Image
        import io
        
        # Create two separate figures to get the rendered images
        # Figure 1: QTEP efficiency base layer
        fig1 = plt.figure(figsize=(12, 6))
        hp.mollview(qtep_map,
                   title='',
                   cmap='viridis',
                   cbar=False,
                   fig=fig1.number)
        
        # Save to buffer
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        buf1.seek(0)
        qtep_img = Image.open(buf1)
        plt.close(fig1)
        
        # Figure 2: CMB temperature layer
        fig2 = plt.figure(figsize=(12, 6))
        hp.mollview(cmb_map,
                   title='',
                   cmap='RdBu_r',
                   cbar=False,
                   min=-3*np.std(cmb_map),
                   max=3*np.std(cmb_map),
                   fig=fig2.number)
        
        # Save to buffer
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        buf2.seek(0)
        cmb_img = Image.open(buf2)
        plt.close(fig2)
        
        # Convert to RGBA
        qtep_rgba = qtep_img.convert('RGBA')
        cmb_rgba = cmb_img.convert('RGBA')
        
        # Resize to match if needed
        if qtep_rgba.size != cmb_rgba.size:
            cmb_rgba = cmb_rgba.resize(qtep_rgba.size, Image.Resampling.LANCZOS)
        
        # Create 25% opacity CMB layer
        cmb_array = np.array(cmb_rgba, dtype=np.float32)
        cmb_array[:, :, 3] = 63.75  # 25% opacity (0-255 scale)
        cmb_rgba = Image.fromarray(cmb_array.astype(np.uint8), 'RGBA')
        
        # Composite: CMB on top of QTEP
        overlay = Image.alpha_composite(qtep_rgba, cmb_rgba)
        
        # Create final figure with overlay and colorbars
        fig = plt.figure(figsize=(16, 6))
        
        # Main overlay
        ax_main = plt.subplot(1, 1, 1)
        ax_main.imshow(overlay)
        ax_main.axis('off')
        ax_main.set_title('CMB Temperature (25% opacity) Overlaid on QTEP Efficiency',
                         fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbars as separate elements
        # QTEP colorbar
        ax_cbar_qtep = fig.add_axes([0.15, 0.12, 0.25, 0.02])
        norm_qtep = Normalize(vmin=np.min(qtep_map), vmax=np.max(qtep_map))
        cb_qtep = plt.colorbar(plt.cm.ScalarMappable(norm=norm_qtep, cmap='viridis'),
                              cax=ax_cbar_qtep, orientation='horizontal')
        cb_qtep.set_label('QTEP Efficiency η (base layer)', fontsize=10)
        
        # CMB colorbar
        ax_cbar_cmb = fig.add_axes([0.60, 0.12, 0.25, 0.02])
        norm_cmb = Normalize(vmin=-3*np.std(cmb_map), vmax=3*np.std(cmb_map))
        cb_cmb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_cmb, cmap='RdBu_r'),
                             cax=ax_cbar_cmb, orientation='horizontal')
        cb_cmb.set_label('CMB Temperature [μK] (25% overlay)', fontsize=10)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cleanup
        buf1.close()
        buf2.close()
        
        logger.info(f"Saved CMB-QTEP overlay to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create overlay figure: {e}")
        import traceback
        logger.warning(traceback.format_exc())


def plot_cross_survey_comparison(survey_results: Dict[str, Dict],
                                output_path: Path) -> None:
    """
    Plot results across multiple CMB datasets.
    
    Parameters:
        survey_results: Dictionary mapping survey names to results
        output_path: Path to save figure
    """
    try:
        surveys = list(survey_results.keys())
        deficits = [survey_results[s]['deficit'] for s in surveys]
        uncertainties = [survey_results[s]['uncertainty'] for s in surveys]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(surveys))
        ax.errorbar(x_pos, deficits, yerr=uncertainties, 
                   fmt='o', capsize=5, capthick=2, linewidth=2,
                   markersize=8, label='Temperature Deficit')
        
        # Add mean line
        mean_deficit = np.mean(deficits)
        ax.axhline(mean_deficit, color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_deficit:.2e}')
        
        ax.set_xlabel('CMB Survey')
        ax.set_ylabel('Temperature Deficit δT/T')
        ax.set_title('Cross-Survey Cold Spot Temperature Deficit Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(surveys, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cross-survey comparison to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create cross-survey comparison figure: {e}")

