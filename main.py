#!/usr/bin/env python3
"""
H-ΛCDM Analysis Framework - Main Entry Point
============================================

Command-line interface for the Holographic Lambda Model (H-ΛCDM) analysis.

This framework tests predictions of the H-ΛCDM model against cosmological
observations using information-theoretic first principles.

Usage:
    python main.py --gamma validate extended
    python main.py --bao --cmb validate --void validate extended
    python main.py --all validate

For detailed help:
    python main.py --help
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Configure root logger to output to console
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from hlcdm.parameters import HLCDM_PARAMS
from pipeline.gamma import GammaPipeline
from pipeline.bao import BAOPipeline
from pipeline.cmb import CMBPipeline
from pipeline.void import VoidPipeline
from pipeline.voidfinder import VoidFinderPipeline
from pipeline.hlcdm import HLCDMPipeline
from pipeline.ml import MLPipeline
from pipeline.common.reporting import HLambdaDMReporter
from pipeline.common.visualization import HLambdaDMVisualizer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='H-ΛCDM Analysis Framework: Information-theoretic cosmology testing',
        epilog=f'Framework v{HLCDM_PARAMS.version} - {HLCDM_PARAMS.paper}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'H-ΛCDM Framework v{HLCDM_PARAMS.version}'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages (errors still shown)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analysis pipelines'
    )

    # Pipeline flags with nargs='*' to accept validation arguments
    parser.add_argument(
        '--gamma',
        nargs='*',
        metavar='VALIDATION',
        help='Run gamma analysis (γ(z), Λ(z)). '
             'Optional arguments: validate, extended. '
             'Example: --gamma validate extended'
    )

    parser.add_argument(
        '--bao',
        nargs='*',
        metavar='VALIDATION',
        help='Run BAO analysis (α predictions). '
             'Optional arguments: validate, extended. '
             'Example: --bao validate'
    )

    parser.add_argument(
        '--cmb',
        nargs='*',
        metavar='VALIDATION',
        help='Run CMB analysis (E-mode signatures). '
             'Optional arguments: validate, extended. '
             'Example: --cmb validate extended'
    )

    parser.add_argument(
        '--void',
        nargs='*',
        metavar='VALIDATION',
        help='Run void analysis (E8×E8 alignments). '
             'Optional arguments: validate, extended. '
             'Example: --void validate extended'
    )

    parser.add_argument(
        '--voidfinder',
        nargs='*',
        metavar='VALIDATION',
        help='Run VoidFinder pipeline (generate void catalogs from galaxy surveys). '
             'Optional arguments: validate, extended. '
             'Example: --voidfinder validate'
    )

    parser.add_argument(
        '--voidfinder-catalog',
        type=str,
        default='sdss_dr16',
        choices=['sdss_dr16', 'sdss_dr7'],
        help='Galaxy catalog to use for void finding (default: sdss_dr16). Available: sdss_dr16, sdss_dr7'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Batch size for processing galaxies (default: 100000)'
    )

    parser.add_argument(
        '--abs-mag-limit',
        type=float,
        default=-20.0,
        help='Absolute magnitude limit for volume-limited sample (default: -20.0)'
    )

    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Number of CPUs for VAST parallelization (default: None = use all physical cores)'
    )

    parser.add_argument(
        '--save-after',
        type=int,
        default=10000,
        help='Save VAST checkpoint after every N cells processed (default: 10000 = enabled)'
    )

    parser.add_argument(
        '--use-start-checkpoint',
        action='store_true',
        help='Resume from VAST checkpoint file if found (default: False)'
    )

    parser.add_argument(
        '--grid-size',
        type=float,
        default=50.0,
        help='VAST grid edge length in Mpc (default: 50.0 Mpc)'
    )

    parser.add_argument(
        '--z-bin-size',
        type=float,
        default=None,
        help='Process in redshift bins of this size (e.g., 0.05 for bins of 0.05 redshift width). None = no binning (default: None)'
    )

    parser.add_argument(
        '--voidfinder-algorithm',
        type=str,
        choices=['vast', 'zobov'],
        default='vast',
        help='Void finding algorithm: vast (sphere-growing) or zobov (Voronoi watershed) (default: vast)'
    )

    parser.add_argument(
        '--zobov-output-name',
        type=str,
        default=None,
        help='MANDATORY for H-ZOBOV: Base name for output files (JSON, catalog, log, report). Format: name-zmin_val-zmax_val-params'
    )

    parser.add_argument(
        '--zobov-batch-size',
        type=int,
        default=50000,
        help='Batch size for MPS-accelerated operations in H-ZOBOV (default: 50000)'
    )

    parser.add_argument(
        '--zobov-use-hlcdm-lambda',
        action='store_true',
        default=True,
        help='Use H-ΛCDM Lambda(z) for void significance (default: True)'
    )

    parser.add_argument(
        '--zobov-no-hlcdm-lambda',
        action='store_false',
        dest='zobov_use_hlcdm_lambda',
        help='Disable H-ΛCDM Lambda(z) and use constant Lambda'
    )

    parser.add_argument(
        '--zobov-significance-ratio',
        type=float,
        default=None,
        help='Density ratio threshold for zone merging in H-ZOBOV (optional, algorithm is parameter-free)'
    )

    parser.add_argument(
        '--zobov-min-void-volume',
        type=float,
        default=None,
        help='Minimum void volume filter in Mpc³ for H-ZOBOV (optional)'
    )

    parser.add_argument(
        '--zobov-z-bin-size',
        type=float,
        default=0.05,
        help='Redshift bin size for H-ZOBOV processing (default: 0.05). Redshift binning is default for H-ZOBOV to ensure consistent Λ(z) within bins. Set to None to disable.'
    )

    parser.add_argument(
        '--hlcdm',
        nargs='*',
        metavar='VALIDATION',
        help='Run H-ΛCDM extension tests (JWST, Lyman-α, FRB, E8 chiral). '
             'Optional arguments: validate, extended. '
             'Example: --hlcdm validate extended'
    )

    parser.add_argument(
        '--ml',
        nargs='*',
        metavar='STAGE',
        help='Run complete 5-stage ML pipeline. '
             'Optional arguments: train, detect, interpret, validate, blind, unblind. '
             'No arguments = run all stages. '
             'Examples: --ml train (Stage 1-2 only), --ml detect (Stage 3 only), --ml (all stages)'
    )

    parser.add_argument(
        '--ml-train',
        action='store_true',
        help='Train SSL encoders + domain adaptation (Stages 1-2)'
    )

    parser.add_argument(
        '--ml-detect',
        action='store_true',
        help='Run blind pattern detection (Stage 3)'
    )

    parser.add_argument(
        '--ml-interpret',
        action='store_true',
        help='Generate LIME/SHAP explanations (Stage 4)'
    )

    parser.add_argument(
        '--ml-validate',
        action='store_true',
        help='Full validation suite (bootstrap, null hypothesis, cross-survey) (Stage 5)'
    )

    parser.add_argument(
        '--ml-blind',
        action='store_true',
        help='Enable blind analysis protocol'
    )

    parser.add_argument(
        '--ml-unblind',
        action='store_true',
        help='Generate unblinding report comparing to H-ΛCDM predictions'
    )

    # Analysis parameters
    parser.add_argument(
        '--z-min',
        type=float,
        default=0.0,
        help='Minimum redshift for analysis (default: 0.0)'
    )

    parser.add_argument(
        '--z-max',
        type=float,
        default=10.0,
        help='Maximum redshift for analysis (default: 10.0)'
    )

    parser.add_argument(
        '--z-steps',
        type=int,
        default=100,
        help='Number of redshift steps (default: 100)'
    )

    parser.add_argument(
        '--bao-datasets',
        nargs='+',
        default=None,  # None means use all available datasets
        choices=['boss_dr12', 'desi', 'desi_y1', 'eboss', 'sixdfgs', 'wigglez',
                 'sdss_mgs', 'sdss_dr7', '2dfgrs', 'des_y1', 'des_y3'],
        help='BAO datasets to analyze (default: all available datasets). Available: '
             'boss_dr12, desi, desi_y1, eboss, sixdfgs, wigglez, '
             'sdss_mgs, sdss_dr7, 2dfgrs, des_y1, des_y3'
    )

    parser.add_argument(
        '--cmb-methods',
        nargs='+',
        default=['wavelet'],
        choices=['wavelet', 'bispectrum', 'topological', 'phase', 'void', 'scale', 'gw', 'ionization', 'isotropy', 'zeno', 'all'],
        help='CMB analysis methods (default: wavelet)'
    )

    parser.add_argument(
        '--cmb-datasets',
        nargs='+',
        default=['act_dr6', 'planck_2018', 'spt3g'],
        choices=['act_dr6', 'planck_2018', 'spt3g'],
        help='CMB datasets to analyze (default: act_dr6 planck_2018 spt3g)'
    )

    parser.add_argument(
        '--void-surveys',
        nargs='+',
        default=['sdss_dr7_douglass', 'sdss_dr7_clampitt', 'desi'],
        choices=['sdss_dr7_douglass', 'sdss_dr7_clampitt', 'desi', 'vide_public'],
        help='Void surveys to analyze. vide_public requires manual download from cosmicvoids.net'
    )

    parser.add_argument(
        '--void-mode',
        type=str,
        choices=['lcdm', 'hlcdm'],
        default='lcdm',
        help='Void analysis mode: lcdm (traditional surveys) or hlcdm (H-ZOBOV catalogs)'
    )

    parser.add_argument(
        '--validation-level',
        choices=['basic', 'extended'],
        default='basic',
        help='Default validation level if not specified per pipeline (default: basic)'
    )

    # Reporting options
    parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate comprehensive analysis report (default: enabled)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_false',
        dest='generate_report',
        help='Disable report generation'
    )
    
    # Set default to True for generate_report if not explicitly set
    parser.set_defaults(generate_report=True)

    parser.add_argument(
        '--generate-figures',
        action='store_true',
        help='Generate publication-quality figures'
    )

    parser.add_argument(
        '--skip-reporting',
        action='store_true',
        help='Skip all reporting and figure generation'
    )

    return parser.parse_args()


def determine_pipeline_config(args) -> Dict[str, Dict[str, Any]]:
    """
    Determine which pipelines to run and their configurations.

    Parameters:
        args: Parsed command-line arguments

    Returns:
        dict: Pipeline configurations
    """
    pipeline_config = {}

    # Handle --all flag
    if args.all:
        pipelines_to_run = ['gamma', 'bao', 'cmb', 'void', 'hlcdm']
        default_validation = ['validate'] if args.validation_level == 'extended' else []
        if args.validation_level == 'extended':
            default_validation.append('extended')
    else:
        pipelines_to_run = []

    # Process individual pipeline flags
    pipeline_flags = {
        'gamma': args.gamma,
        'bao': args.bao,
        'cmb': args.cmb,
        'void': args.void,
        'voidfinder': args.voidfinder,
        'hlcdm': args.hlcdm,
        'ml': args.ml
    }

    # Handle ML subcommands
    ml_subcommands = []
    if args.ml_train:
        ml_subcommands.append('train')
    if args.ml_detect:
        ml_subcommands.append('detect')
    if args.ml_interpret:
        ml_subcommands.append('interpret')
    if args.ml_validate:
        ml_subcommands.append('validate')
    if args.ml_blind:
        ml_subcommands.append('blind')
    if args.ml_unblind:
        ml_subcommands.append('unblind')

    # If ML subcommands specified, override general ML flag
    if ml_subcommands:
        pipeline_flags['ml'] = ml_subcommands

    for pipeline_name, validation_args in pipeline_flags.items():
        if validation_args is not None or (args.all and pipeline_name in pipelines_to_run):
            if pipeline_name not in pipelines_to_run:
                pipelines_to_run.append(pipeline_name)

            # Determine validation level
            if args.all:
                validation_level = default_validation
            else:
                validation_level = validation_args or []

            # Parse validation arguments
            validate_basic = 'validate' in validation_level
            validate_extended = 'extended' in validation_level

            pipeline_config[pipeline_name] = {
                'run': True,
                'validate_basic': validate_basic,
                'validate_extended': validate_extended,
                'context': {}
            }

            # Add pipeline-specific parameters
            if pipeline_name == 'gamma':
                pipeline_config[pipeline_name]['context'].update({
                    'z_min': args.z_min,
                    'z_max': args.z_max,
                    'z_steps': args.z_steps
                })
            elif pipeline_name == 'bao':
                # Use all available datasets if none specified
                bao_datasets = args.bao_datasets if args.bao_datasets else None
                pipeline_config[pipeline_name]['context'].update({
                    'datasets': bao_datasets  # None means use all in pipeline
                })
            elif pipeline_name == 'ml':
                # Handle ML subcommands
                stages = []
                if args.ml_train or not any([args.ml_detect, args.ml_interpret, args.ml_validate, args.ml_blind, args.ml_unblind]):
                    stages.extend(['ssl', 'domain'])  # Train includes SSL + domain adaptation
                if args.ml_detect or not any([args.ml_train, args.ml_interpret, args.ml_validate, args.ml_blind, args.ml_unblind]):
                    stages.append('detect')
                if args.ml_interpret or not any([args.ml_train, args.ml_detect, args.ml_validate, args.ml_blind, args.ml_unblind]):
                    stages.append('interpret')
                if args.ml_validate or not any([args.ml_train, args.ml_detect, args.ml_interpret, args.ml_blind, args.ml_unblind]):
                    stages.append('validate')
                if args.ml_blind:
                    stages.append('blind')
                if args.ml_unblind:
                    stages.append('unblind')

                # If no specific stages requested, run all
                if not stages:
                    stages = ['all']

                pipeline_config[pipeline_name]['context'].update({
                    'stages': stages
                })

            elif pipeline_name == 'cmb':
                methods = args.cmb_methods
                if 'all' in methods:
                    methods = ['wavelet', 'bispectrum', 'topological', 'phase', 'void', 'scale', 'gw', 'ionization', 'isotropy', 'zeno']
                pipeline_config[pipeline_name]['context'].update({
                    'methods': methods
                })
            elif pipeline_name == 'void':
                pipeline_config[pipeline_name]['context'].update({
                    'surveys': args.void_surveys,
                    'mode': args.void_mode
                })
            elif pipeline_name == 'voidfinder':
                # Use algorithm-specific bin size parameter
                z_bin_size = args.zobov_z_bin_size if args.voidfinder_algorithm == 'zobov' else args.z_bin_size
                
                pipeline_config[pipeline_name]['context'].update({
                    'algorithm': args.voidfinder_algorithm,
                    'catalog': args.voidfinder_catalog,
                    'z_min': args.z_min,
                    'z_max': args.z_max,
                    'chunk_size': args.chunk_size,
                    'abs_mag_limit': args.abs_mag_limit,
                    'num_cpus': args.num_cpus,
                    'save_after': args.save_after,
                    'use_start_checkpoint': args.use_start_checkpoint,
                    'grid_size': args.grid_size,
                    'z_bin_size': z_bin_size,
                    # H-ZOBOV specific parameters
                    'output_name': args.zobov_output_name,
                    'batch_size': args.zobov_batch_size,
                    'use_hlcdm_lambda': args.zobov_use_hlcdm_lambda,
                    'significance_ratio': args.zobov_significance_ratio,
                    'min_void_volume': args.zobov_min_void_volume,
                })
                
                # Validate H-ZOBOV requirements
                if args.voidfinder_algorithm == 'zobov':
                    if args.zobov_output_name is None:
                        raise ValueError("--zobov-output-name is MANDATORY when --voidfinder-algorithm zobov")
            elif pipeline_name == 'cmb':
                pipeline_config[pipeline_name]['context'].update({
                    'datasets': args.cmb_datasets
                })

    return {
        'pipelines_to_run': pipelines_to_run,
        'pipeline_config': pipeline_config,
        'reporting': {
            # Report generation is on by default unless explicitly disabled
            'generate_report': (args.generate_report if hasattr(args, 'generate_report') else True) and not args.skip_reporting,
            'generate_figures': args.generate_figures and not args.skip_reporting,
            'skip_reporting': args.skip_reporting
        }
    }


def initialize_pipelines(output_dir: str) -> Dict[str, Any]:
    """
    Initialize all analysis pipelines.

    Parameters:
        output_dir: Output directory

    Returns:
        dict: Initialized pipeline objects
    """
    pipelines = {
        'gamma': GammaPipeline(output_dir),
        'bao': BAOPipeline(output_dir),
        'cmb': CMBPipeline(output_dir),
        'void': VoidPipeline(output_dir),
        'voidfinder': VoidFinderPipeline(output_dir),
        'hlcdm': HLCDMPipeline(output_dir),
        'ml': MLPipeline(output_dir)
    }

    return pipelines


def run_pipeline_analysis(pipeline_name: str, pipeline_obj, config: Dict[str, Any],
                         quiet: bool = False) -> Dict[str, Any]:
    """
    Run analysis for a single pipeline.

    Parameters:
        pipeline_name: Name of the pipeline
        pipeline_obj: Pipeline object
        config: Pipeline configuration
        quiet: Suppress output

    Returns:
        dict: Pipeline results
    """
    import json  # Import at function level to ensure it's always available
    
    if not quiet:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING {pipeline_name.upper()} PIPELINE")
        logger.info(f"{'='*60}")

    results = {}

    # Determine filename based on pipeline mode (for void pipeline)
    context = config.get('context', {})
    mode = context.get('mode', 'lcdm') if context else 'lcdm'
    
    # For void pipeline in H-LCDM mode, use HLCDM_ prefix
    if pipeline_name == 'void' and mode == 'hlcdm':
        base_filename = f"HLCDM_{pipeline_name}_results.json"
        extended_filename = f"HLCDM_{pipeline_name}_results_extended.json"
    else:
        base_filename = f"{pipeline_name}_results.json"
        extended_filename = f"{pipeline_name}_results_extended.json"
    
    # Data persistence paths
    json_path = Path("results") / "json" / base_filename
    extended_json_path = Path("results") / "json" / extended_filename
    
    # Check what data products already exist
    main_results_exist = False
    basic_validation_exists = False
    extended_validation_exists = False
    
    # Check for main results
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            saved_results = saved_data.get('results', {})
            if saved_results.get('main') or saved_results.get('clustering_analysis'):
                main_results_exist = True
                results = saved_results
                if not quiet:
                    logger.info(f"✓ Found existing main analysis results")
            if saved_results.get('validation'):
                basic_validation_exists = True
                if not quiet:
                    logger.info(f"✓ Found existing basic validation results")
        except Exception as e:
            if not quiet:
                logger.warning(f"⚠ Could not load {json_path}: {e}")
    
    # Check for extended validation results
    if extended_json_path.exists() and extended_json_path.stat().st_size > 0:
        try:
            with open(extended_json_path, 'r') as f:
                extended_data = json.load(f)
            extended_results = extended_data.get('results', {}).get('validation_extended', {})
            if extended_results:
                extended_validation_exists = True
                results['validation_extended'] = _convert_json_booleans(extended_results)
                if not quiet:
                    logger.info(f"✓ Found existing extended validation results")
        except Exception as e:
            if not quiet:
                logger.warning(f"⚠ Could not load {extended_json_path}: {e}")
    
    # Determine what needs to run
    need_main_analysis = not main_results_exist
    need_basic_validation = config.get('validate_basic', False) and not basic_validation_exists
    need_extended_validation = config.get('validate_extended', False) and not extended_validation_exists
    
    # If all requested data products exist, skip to reporting
    if not need_main_analysis and not need_basic_validation and not need_extended_validation:
        if not quiet:
            logger.info(f"All requested data products exist for {pipeline_name}, skipping to reporting...")
        return results
    
    # Run only what's needed
    if need_main_analysis or need_basic_validation or need_extended_validation:
        try:
            # Run main analysis if needed
            if need_main_analysis:
                if not quiet:
                    logger.info(f"Executing {pipeline_name} analysis...")
                main_results = pipeline_obj.run(config.get('context', {}))
                # Merge main results into results dict
                if isinstance(main_results, dict):
                    results.update(main_results)
                results['main'] = main_results

            # Run basic validation if needed
            if need_basic_validation:
                if not quiet:
                    logger.info(f"Running basic validation for {pipeline_name}...")
                results['validation'] = pipeline_obj.validate(config.get('context', {}))

            # Run extended validation if needed
            if need_extended_validation:
                if not quiet:
                    logger.info(f"Running extended validation for {pipeline_name}...")
                results['validation_extended'] = pipeline_obj.validate_extended(config.get('context', {}))

            # Save results after each stage completes
            # Save main results to JSON
            try:
                existing_data = {}
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            existing_data = json.load(f)
                    except json.JSONDecodeError as json_err:
                        # If JSON file is malformed, log warning and start fresh
                        logger.warning(f"Existing JSON file {json_path} is malformed (error: {json_err}), starting fresh")
                        existing_data = {}

                if 'results' not in existing_data:
                    existing_data['results'] = {}
                
                # Save all results except extended validation (which has its own file)
                basic_results = {k: v for k, v in results.items() if k != 'validation_extended'}
                existing_data['results'].update(basic_results)
                existing_data = _sanitize_for_json(existing_data)

                with open(json_path, 'w') as f:
                    json.dump(existing_data, f, indent=2, default=str)

                if not quiet:
                    logger.info(f"✓ Results saved: {json_path}")

            except Exception as e:
                logger.warning(f"Could not save results: {e}")

            # Save extended validation to separate file if it was run
            if 'validation_extended' in results:
                try:
                    extended_data = {
                        'pipeline': pipeline_name,
                        'timestamp': int(time.time()),
                        'results': {
                            'validation_extended': results['validation_extended']
                        },
                        'metadata': {
                            'description': f'Extended validation results for {pipeline_name} pipeline',
                            'validation_type': 'extended',
                            'generated_from_main_results': f'{pipeline_name}_results.json'
                        }
                    }
                    extended_data = _sanitize_for_json(extended_data)
                    
                    with open(extended_json_path, 'w') as f:
                        json.dump(extended_data, f, indent=2, default=str)

                    if not quiet:
                        logger.info(f"✓ Extended validation saved: {extended_json_path}")

                except Exception as e:
                    logger.error(f"Could not save extended validation results: {e}")

        except Exception as e:
            logger.error(f"✗ Error in {pipeline_name} pipeline: {e}")
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()

    return results


def _sanitize_for_json(obj):
    """
    Sanitize data for JSON serialization by converting inf/nan to None.
    Also converts numpy types in keys and values to Python native types.
    
    Parameters:
        obj: Object to sanitize
        
    Returns:
        Sanitized object safe for JSON serialization
    """
    import math
    import numpy as np
    
    if isinstance(obj, dict):
        # Convert numpy keys to strings/ints, and sanitize values
        sanitized_dict = {}
        for k, v in obj.items():
            # Convert numpy key types to Python native types
            if isinstance(k, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                sanitized_key = int(k)
            elif isinstance(k, (np.floating, np.float16, np.float32, np.float64)):
                sanitized_key = float(k)
            elif isinstance(k, (np.bool_, bool)):
                sanitized_key = bool(k)
            elif isinstance(k, str):
                sanitized_key = k
            else:
                # Fallback: convert to string
                sanitized_key = str(k)
            sanitized_dict[sanitized_key] = _sanitize_for_json(v)
        return sanitized_dict
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_json(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        val = float(obj)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj


def _convert_json_booleans(obj):
    """Convert string representations of booleans to actual booleans."""
    if isinstance(obj, dict):
        return {k: _convert_json_booleans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_json_booleans(item) for item in obj]
    elif isinstance(obj, str):
        if obj.lower() == 'true':
            return True
        elif obj.lower() == 'false':
            return False
    return obj


def generate_reports(all_results: Dict[str, Any], output_dir: str,
                    generate_report: bool = True, generate_figures: bool = True,
                    quiet: bool = False):
    """
    Generate comprehensive reports and figures.

    Parameters:
        all_results: Results from all pipelines
        output_dir: Output directory
        generate_report: Generate text report
        generate_figures: Generate figures
        quiet: Suppress output
    """
    if not quiet:
        logger.info(f"\n{'='*60}")
        logger.info("GENERATING REPORTS AND FIGURES")
        logger.info(f"{'='*60}")

    reporter = HLambdaDMReporter(output_dir)
    visualizer = HLambdaDMVisualizer(output_dir)

    generated_files = []

    try:
        # Generate comprehensive report
        if generate_report:
            if not quiet:
                logger.info("Generating comprehensive analysis report...")
            report_path = reporter.generate_comprehensive_report(all_results)
            generated_files.append(report_path)
            if not quiet:
                logger.info(f"✓ Report generated: {report_path}")

        # Generate comprehensive figure
        if generate_figures:
            if not quiet:
                logger.info("Generating comprehensive analysis figures...")
            figure_path = visualizer.create_comprehensive_figure(all_results)
            generated_files.append(figure_path)

            # Generate validation figure
            validation_figure = visualizer.create_validation_figure(all_results)
            generated_files.append(validation_figure)

            if not quiet:
                logger.info(f"✓ Figures generated: {figure_path}")
                logger.info(f"✓ Validation figure: {validation_figure}")

        # Generate individual pipeline reports
        if generate_report:
            for pipeline_name, results in all_results.items():
                if not quiet:
                    logger.info(f"Generating {pipeline_name} pipeline report...")
                pipeline_report = reporter.generate_pipeline_report(pipeline_name, results)
                generated_files.append(pipeline_report)

                if not quiet:
                    logger.info(f"✓ {pipeline_name.upper()} report: {pipeline_report}")

        # Generate individual pipeline figures
        if generate_figures:
            for pipeline_name, results in all_results.items():
                if not quiet:
                    logger.info(f"Generating {pipeline_name} pipeline figures...")
                pipeline_figure = visualizer.create_pipeline_figure(pipeline_name, results)
                generated_files.append(pipeline_figure)

                if not quiet:
                    logger.info(f"✓ {pipeline_name.upper()} figures: {pipeline_figure}")

    except Exception as e:
        logger.error(f"✗ Error generating reports: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    return generated_files


def save_execution_summary(all_results: Dict[str, Any], config: Dict[str, Any],
                          generated_files: List[str], output_dir: str,
                          execution_time: float):
    """
    Save execution summary.

    Parameters:
        all_results: Pipeline results
        config: Execution configuration
        generated_files: Generated report/figure files
        output_dir: Output directory
        execution_time: Total execution time
    """
    summary = {
        'execution_summary': {
            'timestamp': time.time(),
            'execution_time_seconds': execution_time,
            'pipelines_run': config['pipelines_to_run'],
            'pipeline_config': config['pipeline_config'],
            'reporting_config': config['reporting'],
            'generated_files': generated_files,
            'results_summary': {}
        }
    }

    # Add results summaries
    for pipeline_name, results in all_results.items():
        summary['execution_summary']['results_summary'][pipeline_name] = {
            'completed': 'error' not in results,
            'has_validation': 'validation' in results,
            'has_extended_validation': 'validation_extended' in results
        }

    # Save summary
    summary_path = Path(output_dir) / "execution_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n✓ Execution summary saved: {summary_path}")


def main():
    """
    Main entry point for H-ΛCDM analysis framework.
    """
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Determine pipeline configuration
    config = determine_pipeline_config(args)

    if not config['pipelines_to_run']:
        logger.warning("No pipelines selected. Use --help for usage information.")
        logger.info("Example: python main.py --gamma validate")
        return 1

    # Initialize output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    if not args.quiet:
        logger.info(f"{'='*70}")
        logger.info("H-ΛCDM ANALYSIS FRAMEWORK")
        logger.info(f"Version {HLCDM_PARAMS.version}")
        logger.info(HLCDM_PARAMS.paper)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"{'='*70}")
        logger.info(f"Pipelines to run: {', '.join(config['pipelines_to_run'])}")
        logger.info(f"{'='*70}\n")

    # Initialize pipelines
    pipelines = initialize_pipelines(str(output_dir))

    # Run pipeline analyses
    all_results = {}
    for pipeline_name in config['pipelines_to_run']:
        pipeline_config = config['pipeline_config'][pipeline_name]
        pipeline_obj = pipelines[pipeline_name]

        results = run_pipeline_analysis(
            pipeline_name, pipeline_obj, pipeline_config, args.quiet
        )

        all_results[pipeline_name] = results

    # Generate reports and figures (unless skipped)
    generated_files = []
    if not config['reporting']['skip_reporting']:
        generated_files = generate_reports(
            all_results, str(output_dir),
            config['reporting']['generate_report'],
            config['reporting']['generate_figures'],
            args.quiet
        )
    elif not args.quiet:
        logger.info("Skipping report and figure generation (--skip-reporting)")

    # Calculate execution time
    execution_time = time.time() - start_time

    # Save execution summary
    save_execution_summary(all_results, config, generated_files,
                          str(output_dir), execution_time)

    # Print completion summary
    if not args.quiet:
        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total execution time: {execution_time:.1f} seconds")
        logger.info(f"Pipelines completed: {len(config['pipelines_to_run'])}")
        logger.info(f"Results saved to: {args.output_dir}")
        if generated_files:
            logger.info(f"Generated files: {len(generated_files)}")
            for file_path in generated_files[:5]:  # Show first 5
                logger.info(f"  - {file_path}")
            if len(generated_files) > 5:
                logger.info(f"  ... and {len(generated_files) - 5} more")
        logger.info(f"{'='*70}")

    # Return success/failure status
    success = all('error' not in results for results in all_results.values())
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
