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
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from hlcdm.parameters import HLCDM_PARAMS
from pipeline.gamma import GammaPipeline
from pipeline.bao import BAOPipeline
from pipeline.cmb import CMBPipeline
from pipeline.void import VoidPipeline
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
        default=['sdss_dr7_douglass', 'sdss_dr7_clampitt'],
        choices=['sdss_dr7_douglass', 'sdss_dr7_clampitt'],
        help='Void surveys to analyze (default: sdss_dr7_douglass sdss_dr7_clampitt)'
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
                    'surveys': args.void_surveys
                })
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
    if not quiet:
        print(f"\n{'='*60}")
        print(f"RUNNING {pipeline_name.upper()} PIPELINE")
        print(f"{'='*60}")

    results = {}

    try:
        # Run main analysis
        if not quiet:
            print(f"Executing {pipeline_name} analysis...")
        results['main'] = pipeline_obj.run(config.get('context', {}))

        # Run basic validation
        if config.get('validate_basic', False):
            if not quiet:
                print(f"Running basic validation for {pipeline_name}...")
            results['validation'] = pipeline_obj.validate(config.get('context', {}))

        # Run extended validation
        if config.get('validate_extended', False):
            if not quiet:
                print(f"Running extended validation for {pipeline_name}...")
            results['validation_extended'] = pipeline_obj.validate_extended(config.get('context', {}))

        if not quiet:
            print(f"✓ {pipeline_name.upper()} pipeline completed successfully")

    except Exception as e:
        print(f"✗ Error in {pipeline_name} pipeline: {e}")
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    return results


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
        print(f"\n{'='*60}")
        print("GENERATING REPORTS AND FIGURES")
        print(f"{'='*60}")

    reporter = HLambdaDMReporter(output_dir)
    visualizer = HLambdaDMVisualizer(output_dir)

    generated_files = []

    try:
        # Generate comprehensive report
        if generate_report:
            if not quiet:
                print("Generating comprehensive analysis report...")
            report_path = reporter.generate_comprehensive_report(all_results)
            generated_files.append(report_path)
            if not quiet:
                print(f"✓ Report generated: {report_path}")

        # Generate comprehensive figure
        if generate_figures:
            if not quiet:
                print("Generating comprehensive analysis figures...")
            figure_path = visualizer.create_comprehensive_figure(all_results)
            generated_files.append(figure_path)

            # Generate validation figure
            validation_figure = visualizer.create_validation_figure(all_results)
            generated_files.append(validation_figure)

            if not quiet:
                print(f"✓ Figures generated: {figure_path}")
                print(f"✓ Validation figure: {validation_figure}")

        # Generate individual pipeline reports
        if generate_report:
            for pipeline_name, results in all_results.items():
                if not quiet:
                    print(f"Generating {pipeline_name} pipeline report...")
                pipeline_report = reporter.generate_pipeline_report(pipeline_name, results)
                generated_files.append(pipeline_report)

                if not quiet:
                    print(f"✓ {pipeline_name.upper()} report: {pipeline_report}")

        # Generate individual pipeline figures
        if generate_figures:
            for pipeline_name, results in all_results.items():
                if not quiet:
                    print(f"Generating {pipeline_name} pipeline figures...")
                pipeline_figure = visualizer.create_pipeline_figure(pipeline_name, results)
                generated_files.append(pipeline_figure)

                if not quiet:
                    print(f"✓ {pipeline_name.upper()} figures: {pipeline_figure}")

    except Exception as e:
        print(f"✗ Error generating reports: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

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

    print(f"\n✓ Execution summary saved: {summary_path}")


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
        print("No pipelines selected. Use --help for usage information.")
        print("Example: python main.py --gamma validate")
        return 1

    # Initialize output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    if not args.quiet:
        print(f"{'='*70}")
        print("H-ΛCDM ANALYSIS FRAMEWORK")
        print(f"Version {HLCDM_PARAMS.version}")
        print(HLCDM_PARAMS.paper)
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*70}")
        print(f"Pipelines to run: {', '.join(config['pipelines_to_run'])}")
        print(f"{'='*70}\n")

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
        print("Skipping report and figure generation (--skip-reporting)")

    # Calculate execution time
    execution_time = time.time() - start_time

    # Save execution summary
    save_execution_summary(all_results, config, generated_files,
                          str(output_dir), execution_time)

    # Print completion summary
    if not args.quiet:
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Total execution time: {execution_time:.1f} seconds")
        print(f"Pipelines completed: {len(config['pipelines_to_run'])}")
        print(f"Results saved to: {args.output_dir}")
        if generated_files:
            print(f"Generated files: {len(generated_files)}")
            for file_path in generated_files[:5]:  # Show first 5
                print(f"  - {file_path}")
            if len(generated_files) > 5:
                print(f"  ... and {len(generated_files) - 5} more")
        print(f"{'='*70}")

    # Return success/failure status
    success = all('error' not in results for results in all_results.values())
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
