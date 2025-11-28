"""
H-ZOBOV Parameter Configuration
==============================

Parameter configuration for the H-ZOBOV void-finding algorithm.
ZOBOV is parameter-free by design, but H-ZOBOV adds optional parameters
for H-ΛCDM integration and MPS acceleration settings.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class HZOBOVParameters:
    """
    Configuration parameters for H-ZOBOV algorithm.
    
    Most parameters are optional since ZOBOV is parameter-free.
    H-ZOBOV specific parameters control Lambda integration and MPS settings.
    """
    
    # H-ΛCDM Integration Parameters
    use_hlcdm_lambda: bool = True
    """Use redshift-dependent Λ(z) from H-ΛCDM instead of constant Λ."""
    
    # Zone Merging Parameters (optional - algorithm is parameter-free)
    significance_ratio: Optional[float] = None
    """Density ratio threshold for zone merging. None = parameter-free mode."""
    
    min_void_volume: Optional[float] = None
    """Minimum void volume in Mpc³. None = no minimum volume filter."""
    
    # MPS Acceleration Parameters
    batch_size: int = 50000
    """Batch size for MPS-accelerated operations."""
    
    mps_device: Optional[str] = None
    """MPS device name. None = auto-detect."""
    
    # Output Parameters
    output_name: Optional[str] = None
    """Base name for output files (MANDATORY for H-ZOBOV pipeline)."""
    
    # Redshift Range (for Lambda calculations)
    z_min: float = 0.0
    """Minimum redshift for analysis."""
    
    z_max: float = 1.0
    """Maximum redshift for analysis."""
    
    # Redshift Binning (default for H-ZOBOV due to Lambda(z) evolution)
    z_bin_size: Optional[float] = 0.05
    """Redshift bin size for processing. Default 0.05 for H-ZOBOV to ensure consistent Λ(z) within bins."""
    
    def validate(self) -> None:
        """
        Validate parameter configuration.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.z_min < 0:
            raise ValueError(f"z_min must be >= 0, got {self.z_min}")
        
        if self.z_max <= self.z_min:
            raise ValueError(f"z_max ({self.z_max}) must be > z_min ({self.z_min})")
        
        if self.z_bin_size is not None and self.z_bin_size <= 0:
            raise ValueError(f"z_bin_size must be positive, got {self.z_bin_size}")
        
        # Auto-disable binning if range is smaller than bin size
        if self.z_bin_size is not None and self.z_bin_size >= (self.z_max - self.z_min):
            # For small redshift ranges, disable binning automatically
            self.z_bin_size = None
        
        if self.significance_ratio is not None and self.significance_ratio <= 0:
            raise ValueError(f"significance_ratio must be positive, got {self.significance_ratio}")
        
        if self.min_void_volume is not None and self.min_void_volume <= 0:
            raise ValueError(f"min_void_volume must be positive, got {self.min_void_volume}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'use_hlcdm_lambda': self.use_hlcdm_lambda,
            'significance_ratio': self.significance_ratio,
            'min_void_volume': self.min_void_volume,
            'batch_size': self.batch_size,
            'mps_device': self.mps_device,
            'output_name': self.output_name,
            'z_min': self.z_min,
            'z_max': self.z_max,
            'z_bin_size': self.z_bin_size,
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'HZOBOVParameters':
        """Create parameters from dictionary."""
        return cls(**params)
    
    def get_output_filename_base(self, include_params: bool = True) -> str:
        """
        Generate base filename from parameters.
        
        Format: {output_name}-zmin_{z_min}-zmax_{z_max}-{other_params}
        Decimals represented as underscores.
        
        Parameters:
            include_params: Whether to include parameter values in filename
            
        Returns:
            Base filename string
        """
        if self.output_name is None:
            raise ValueError("output_name is required for filename generation")
        
        base = self.output_name
        
        if include_params:
            # Format redshifts with underscores for decimals
            z_min_str = str(self.z_min).replace('.', '_')
            z_max_str = str(self.z_max).replace('.', '_')
            
            parts = [
                base,
                f"zmin_{z_min_str}",
                f"zmax_{z_max_str}",
            ]
            
            if self.significance_ratio is not None:
                sig_str = str(self.significance_ratio).replace('.', '_')
                parts.append(f"sig_{sig_str}")
            
            if self.min_void_volume is not None:
                vol_str = str(self.min_void_volume).replace('.', '_')
                parts.append(f"minvol_{vol_str}")
            
            return "-".join(parts)
        
        return base

