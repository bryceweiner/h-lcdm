#!/usr/bin/env python3
"""
ML Model Training Script
========================

Command-line interface for training ML models in the cosmological pattern detection pipeline.

Usage:
    python scripts/train_ml_models.py --stage ssl --epochs 100 --batch-size 256
    python scripts/train_ml_models.py --stage domain --surveys 5
    python scripts/train_ml_models.py --stage detect --modalities cmb,bao,void
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.ml.ml_pipeline import MLPipeline


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML models for cosmological pattern detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SSL encoders on all data
  python scripts/train_ml_models.py --stage ssl --epochs 100 --batch-size 256

  # Perform domain adaptation
  python scripts/train_ml_models.py --stage domain --surveys 5

  # Train pattern detection ensemble
  python scripts/train_ml_models.py --stage detect --modalities cmb,bao,void

  # Run full pipeline training
  python scripts/train_ml_models.py --stage all --output-dir results/ml_training
        """
    )

    # Stage selection
    parser.add_argument(
        '--stage',
        choices=['ssl', 'domain', 'detect', 'interpret', 'validate', 'all'],
        required=True,
        help='Training stage to run'
    )

    # SSL training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs for SSL (default: 100)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for SSL training (default: 256)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate for SSL training (default: 1e-3)'
    )

    # Domain adaptation parameters
    parser.add_argument(
        '--surveys',
        type=int,
        default=5,
        help='Number of surveys for domain adaptation (default: 5)'
    )

    # Detection parameters
    parser.add_argument(
        '--modalities',
        type=str,
        default='cmb,bao,void,galaxy,frb,lyman_alpha,jwst',
        help='Modalities to include in training (comma-separated)'
    )

    # General parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/ml_training',
        help='Output directory for trained models'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Set random seed
    import torch
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        # Initialize ML pipeline
        logger.info(f"Initializing ML pipeline with output dir: {args.output_dir}")
        pipeline = MLPipeline(output_dir=args.output_dir)

        # Prepare context based on stage
        context = {}

        if args.stage == 'ssl':
            context = {
                'stages': ['ssl'],
                'ssl_config': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr
                }
            }

        elif args.stage == 'domain':
            context = {
                'stages': ['domain'],
                'domain_config': {
                    'n_surveys': args.surveys
                }
            }

        elif args.stage == 'detect':
            modalities = args.modalities.split(',')
            context = {
                'stages': ['detect'],
                'detection_config': {
                    'modalities': modalities
                }
            }

        elif args.stage == 'interpret':
            context = {
                'stages': ['interpret']
            }

        elif args.stage == 'validate':
            context = {
                'stages': ['validate']
            }

        elif args.stage == 'all':
            context = {
                'stages': ['all'],
                'full_config': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'n_surveys': args.surveys,
                    'modalities': args.modalities.split(',')
                }
            }

        # Run training
        logger.info(f"Starting {args.stage} stage training")
        results = pipeline.run(context=context)

        # Save results
        import json
        results_file = Path(args.output_dir) / f"training_results_{args.stage}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Training completed. Results saved to {results_file}")

        # Print summary
        if 'pipeline_completed' in results and results['pipeline_completed']:
            print("✓ Training completed successfully!")
            if 'key_findings' in results:
                print("Key findings:")
                for key, value in results['key_findings'].items():
                    print(f"  - {key}: {value}")
        else:
            print("✗ Training failed or incomplete")
            if 'error' in results:
                print(f"Error: {results['error']}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
