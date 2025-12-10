"""
Checkpoint Manager Module
=========================

Handles saving and loading checkpoints for pipeline stages.
"""

import pickle
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging


class CheckpointManager:
    """
    Manages checkpoint saving and loading for ML pipeline stages.
    
    Separated from main pipeline for better organization and testability.
    """
    
    def __init__(self, checkpoint_dir: Path, logger: logging.Logger):
        """
        Initialize checkpoint manager.
        
        Parameters:
            checkpoint_dir: Directory for storing checkpoints
            logger: Logger instance
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    def save_stage_checkpoint(self, stage_name: str, checkpoint_data: Dict[str, Any], 
                             results: Dict[str, Any]) -> None:
        """
        Save checkpoint for a pipeline stage.
        
        Parameters:
            stage_name: Name of the stage (e.g., 'stage1_ssl_training')
            checkpoint_data: Data to save (models, state, etc.)
            results: Results dictionary to save as JSON
        """
        checkpoint_file = self.checkpoint_dir / f"{stage_name}.pkl"
        results_file = self.checkpoint_dir / f"{stage_name}_results.json"
        
        # Ensure checkpoint directory exists and is writable
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint_dir.exists():
            raise RuntimeError(f"Checkpoint directory does not exist and could not be created: {self.checkpoint_dir}")
        if not os.access(self.checkpoint_dir, os.W_OK):
            raise RuntimeError(f"Checkpoint directory is not writable: {self.checkpoint_dir}")
        
        self.logger.info(f"Attempting to save checkpoint for {stage_name} to {checkpoint_file}")
        self.logger.debug(f"Checkpoint data keys: {list(checkpoint_data.keys())}")
        self.logger.debug(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")
        
        try:
            # Save checkpoint (models, state)
            # Note: PyTorch models are saved directly (they're pickle-able)
            # For large models, consider saving state_dicts separately
            with open(checkpoint_file, 'wb') as f:
                # Filter out non-picklable items and convert to CPU if needed
                checkpoint_save = {}
                for key, value in checkpoint_data.items():
                    if value is None:
                        continue
                    # PyTorch models can be pickled, but move to CPU first to save space
                    if isinstance(value, torch.nn.Module):
                        # Save model on CPU to reduce checkpoint size
                        value_cpu = value.cpu() if hasattr(value, 'cpu') else value
                        checkpoint_save[key] = value_cpu
                    elif isinstance(value, (torch.Tensor,)):
                        # Save tensors on CPU
                        value_cpu = value.cpu() if hasattr(value, 'cpu') else value
                        checkpoint_save[key] = value_cpu
                    elif isinstance(value, (dict, list, str, int, float, bool)):
                        # Basic types are picklable
                        checkpoint_save[key] = value
                    elif hasattr(value, '__dict__'):
                        # Try to pickle objects with __dict__
                        try:
                            pickle.dumps(value)  # Test if picklable
                            checkpoint_save[key] = value
                        except:
                            self.logger.warning(f"Skipping non-picklable object {key} in checkpoint")
                    else:
                        checkpoint_save[key] = value
                
                pickle.dump(checkpoint_save, f)
            
            # Save results as JSON (for human-readable inspection)
            with open(results_file, 'w') as f:
                # Convert numpy arrays and other non-JSON types to lists/strings
                results_json = self._convert_to_json_serializable(results)
                json.dump(results_json, f, indent=2)
            
            # Verify files were created
            if checkpoint_file.exists() and results_file.exists():
                checkpoint_size = checkpoint_file.stat().st_size
                results_size = results_file.stat().st_size
                self.logger.info(f"✓ Successfully saved checkpoint for {stage_name}: "
                               f"checkpoint={checkpoint_size} bytes, results={results_size} bytes")
            else:
                raise RuntimeError(f"Checkpoint files were not created: "
                                 f"checkpoint exists={checkpoint_file.exists()}, "
                                 f"results exists={results_file.exists()}")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to save checkpoint for {stage_name}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise
    
    def load_stage_checkpoint(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a pipeline stage.
        
        Parameters:
            stage_name: Name of the stage
        
        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{stage_name}.pkl"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.logger.info(f"Loaded checkpoint for {stage_name}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for {stage_name}: {e}")
            return None
    
    def load_stage_results(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Load results JSON for a pipeline stage.
        
        Parameters:
            stage_name: Name of the stage
        
        Returns:
            Results dictionary or None if not found
        """
        results_file = self.checkpoint_dir / f"{stage_name}_results.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load results for {stage_name}: {e}")
            return None
    
    def ensure_stage_completed(self, stage_name: str, checkpoint_name: str) -> None:
        """
        Ensure a stage has been completed (checkpoint exists).
        
        Parameters:
            stage_name: Name of the stage
            checkpoint_name: Name of the checkpoint file
        
        Raises:
            ValueError: If checkpoint doesn't exist
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        results_file = self.checkpoint_dir / f"{checkpoint_name}_results.json"
        
        if checkpoint_file.exists() and results_file.exists():
            self.logger.info(f"Stage {stage_name} already completed (checkpoint found)")
        else:
            raise ValueError(f"{stage_name} must be completed before proceeding")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

