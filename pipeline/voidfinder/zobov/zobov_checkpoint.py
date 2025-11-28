"""
H-ZOBOV Checkpoint Management
============================

Stage-based checkpointing for H-ZOBOV algorithm to enable resume capability
for long-running void-finding operations on large datasets.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd

from .zobov_parameters import HZOBOVParameters

# H-ZOBOV processing stages
HZOBOV_STAGES = [
    'voronoi',      # Voronoi tessellation
    'density',      # Density field estimation (DTFE)
    'watershed',    # Watershed zone finding
    'merging',      # Zone merging
    'catalog'       # Void catalog generation
]


class HZOBOVCheckpointError(Exception):
    """Error in checkpoint operations."""
    pass


class HZOBOVCheckpointManager:
    """
    Manages checkpointing for H-ZOBOV algorithm stages.
    
    Provides atomic writes and resume capability for each of the 5 stages.
    """
    
    def __init__(self, checkpoint_dir: Path, output_name: str):
        """
        Initialize checkpoint manager.
        
        Parameters:
            checkpoint_dir: Directory to store checkpoint files
            output_name: Base name for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_name = output_name
        self.checkpoint_prefix = f"hzobov_{output_name}"
    
    def get_checkpoint_path(self, stage: str) -> Path:
        """Get checkpoint file path for a given stage."""
        if stage not in HZOBOV_STAGES:
            raise HZOBOVCheckpointError(f"Unknown stage: {stage}. Valid stages: {HZOBOV_STAGES}")
        return self.checkpoint_dir / f"{self.checkpoint_prefix}_{stage}.json"
    
    def save_checkpoint(self, stage: str, data: Dict[str, Any]) -> Path:
        """
        Save checkpoint data atomically.
        
        Parameters:
            stage: Stage identifier (must be in HZOBOV_STAGES)
            data: Checkpoint data dictionary
            
        Returns:
            Path to saved checkpoint file
            
        Raises:
            HZOBOVCheckpointError: If stage is invalid or save fails
        """
        checkpoint_path = self.get_checkpoint_path(stage)
        
        # Add metadata
        checkpoint_data = {
            'stage': stage,
            'output_name': self.output_name,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'data': data
        }
        
        # Atomic write: write to temp file, then rename
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=self._json_serializer)
            
            # Atomic rename
            temp_path.replace(checkpoint_path)
            
            return checkpoint_path
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise HZOBOVCheckpointError(f"Failed to save checkpoint for stage {stage}: {e}") from e
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data if it exists.
        
        Parameters:
            stage: Stage identifier
            
        Returns:
            Checkpoint data dictionary or None if not found
            
        Raises:
            HZOBOVCheckpointError: If checkpoint is corrupted
        """
        checkpoint_path = self.get_checkpoint_path(stage)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            return checkpoint_data.get('data')
            
        except Exception as e:
            raise HZOBOVCheckpointError(f"Failed to load checkpoint for stage {stage}: {e}") from e
    
    def has_checkpoint(self, stage: str) -> bool:
        """Check if checkpoint exists for a stage."""
        checkpoint_path = self.get_checkpoint_path(stage)
        return checkpoint_path.exists()
    
    def clear_checkpoint(self, stage: str):
        """Clear a checkpoint file."""
        checkpoint_path = self.get_checkpoint_path(stage)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    
    def clear_all_checkpoints(self):
        """Clear all checkpoints for this output name."""
        for stage in HZOBOV_STAGES:
            self.clear_checkpoint(stage)
    
    def get_checkpoint_stage(self) -> Optional[str]:
        """
        Determine the latest completed stage from checkpoints.
        
        Returns:
            Latest completed stage name, or None if no checkpoints exist
        """
        latest_stage = None
        latest_time = 0
        
        for stage in HZOBOV_STAGES:
            checkpoint_path = self.get_checkpoint_path(stage)
            if checkpoint_path.exists():
                try:
                    mtime = checkpoint_path.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_stage = stage
                except Exception:
                    continue
        
        return latest_stage
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

