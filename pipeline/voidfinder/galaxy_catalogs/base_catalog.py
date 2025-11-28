"""
Base Galaxy Catalog Interface
==============================

Abstract base class for galaxy catalog providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd


class BaseGalaxyCatalog(ABC):
    """
    Abstract base class for galaxy catalog providers.
    
    All catalog implementations must provide:
    - download(): Download catalog data
    - load(): Load catalog from cache
    - get_volume_limited_sample(): Get volume-limited subsample
    """
    
    def __init__(self, downloaded_data_dir: Path, processed_data_dir: Path):
        """
        Initialize catalog provider.
        
        Parameters:
            downloaded_data_dir: Directory for raw downloaded data
            processed_data_dir: Directory for processed data
        """
        self.downloaded_data_dir = Path(downloaded_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.downloaded_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def catalog_name(self) -> str:
        """Return catalog identifier name."""
        pass
    
    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """
        Return list of required column names.
        
        Must include: ra, dec, z, magnitude, objid
        """
        pass
    
    @abstractmethod
    def download(self, checkpoint_manager=None, **kwargs) -> pd.DataFrame:
        """
        Download galaxy catalog.
        
        Parameters:
            checkpoint_manager: Optional CheckpointManager for resuming
            **kwargs: Catalog-specific download parameters
            
        Returns:
            DataFrame with galaxy catalog
        """
        pass
    
    @abstractmethod
    def load(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load catalog from cache if available.
        
        Parameters:
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame or None if not cached
        """
        pass
    
    def get_volume_limited_sample(self, df: pd.DataFrame, 
                                  z_max: float = 0.2,
                                  mag_limit: float = 21.0) -> pd.DataFrame:
        """
        Get volume-limited sample from catalog.
        
        Parameters:
            df: Full catalog DataFrame
            z_max: Maximum redshift
            mag_limit: Magnitude limit
            
        Returns:
            Volume-limited sample DataFrame
        """
        mask = (df['z'] <= z_max)
        
        if 'magnitude' in df.columns:
            mask &= (df['magnitude'] <= mag_limit)
        elif 'r_mag' in df.columns:
            mask &= (df['r_mag'] <= mag_limit)
        
        return df[mask].copy()
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Parameters:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

