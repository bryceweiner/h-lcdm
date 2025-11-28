"""
Checkpoint Manager for VoidFinder Pipeline
==========================================

Manages checkpointing for large dataset processing to enable resume capability.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
import shutil


class CheckpointManager:
    """
    Manages checkpointing for download and void finding processes.
    
    Provides atomic writes and resume capability for large dataset processing.
    """
    
    def __init__(self, checkpoint_dir: Path, pipeline_name: str = "voidfinder"):
        """
        Initialize checkpoint manager.
        
        Parameters:
            checkpoint_dir: Directory to store checkpoint files
            pipeline_name: Name prefix for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_name = pipeline_name
        
    def get_checkpoint_path(self, stage: str) -> Path:
        """Get checkpoint file path for a given stage."""
        return self.checkpoint_dir / f"{self.pipeline_name}_{stage}.json"
    
    def save_checkpoint(self, stage: str, data: Dict[str, Any]) -> Path:
        """
        Save checkpoint data atomically.
        
        Parameters:
            stage: Stage identifier (e.g., 'download', 'voidfinding')
            data: Checkpoint data dictionary
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = self.get_checkpoint_path(stage)
        
        # Add timestamp
        data['timestamp'] = time.time()
        data['datetime'] = datetime.now().isoformat()
        
        # Atomic write: write to temp file, then rename
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Atomic rename (works on Unix and Windows)
            temp_path.replace(checkpoint_path)
            
            return checkpoint_path
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data if it exists.
        
        Parameters:
            stage: Stage identifier
            
        Returns:
            Checkpoint data dictionary or None if not found
        """
        checkpoint_path = self.get_checkpoint_path(stage)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            # If checkpoint is corrupted, return None (will restart)
            return None
    
    def clear_checkpoint(self, stage: str):
        """Clear a checkpoint file."""
        checkpoint_path = self.get_checkpoint_path(stage)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    
    def has_checkpoint(self, stage: str) -> bool:
        """Check if checkpoint exists for a stage."""
        return self.get_checkpoint_path(stage).exists()
    
    def update_download_progress(self, batch_id: int, total_batches: int, 
                                 rows_downloaded: int, catalog_name: str, 
                                 last_objid: int = 0) -> Path:
        """
        Update download checkpoint.
        
        Parameters:
            batch_id: Current batch ID (0-indexed)
            total_batches: Total number of batches (may be updated dynamically)
            rows_downloaded: Total rows downloaded so far
            catalog_name: Name of catalog being downloaded
            last_objid: Last objid downloaded (for resume without duplicates)
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            'stage': 'download',
            'catalog_name': catalog_name,
            'batch_id': batch_id,
            'total_batches': total_batches,
            'rows_downloaded': rows_downloaded,
            'last_objid': last_objid,
            'progress': batch_id / total_batches if total_batches > 0 else 0.0,
            'completed': False  # Will be set to True when download completes
        }
        return self.save_checkpoint('download', checkpoint_data)
    
    def update_voidfinding_progress(self, chunk_id: int, total_chunks: int,
                                   voids_found: int, parameters: Dict[str, Any]) -> Path:
        """
        Update void finding checkpoint.
        
        Parameters:
            chunk_id: Current chunk ID (0-indexed)
            total_chunks: Total number of chunks
            voids_found: Total voids found so far
            parameters: VoidFinder parameters used
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            'stage': 'voidfinding',
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'voids_found': voids_found,
            'parameters': parameters,
            'progress': chunk_id / total_chunks if total_chunks > 0 else 0.0,
            'completed': chunk_id >= total_chunks
        }
        return self.save_checkpoint('voidfinding', checkpoint_data)

