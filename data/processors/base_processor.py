"""
Base Data Processor
===================

Base class for data processing in H-ΛCDM analysis.

Provides common functionality for:
- Data validation
- Caching of processed data
- Progress logging
- Error handling
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import hashlib


class BaseDataProcessor(ABC):
    """
    Base class for data processors in H-ΛCDM analysis.

    Provides common functionality for data processing, caching, and validation.
    """

    def __init__(self, downloaded_data_dir: str = "downloaded_data",
                 processed_data_dir: str = "processed_data"):
        """
        Initialize base processor.

        Parameters:
            downloaded_data_dir (str): Directory with raw downloaded data
            processed_data_dir (str): Directory for processed data output
        """
        self.downloaded_data_dir = Path(downloaded_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def get_processed_data_path(self, dataset_name: str, suffix: str = ".pkl") -> Path:
        """
        Get path for processed data file.

        Parameters:
            dataset_name (str): Name of the dataset
            suffix (str): File suffix (default: .pkl)

        Returns:
            Path: Path to processed data file
        """
        return self.processed_data_dir / f"{dataset_name}_processed{suffix}"

    def is_processed_data_fresh(self, dataset_name: str, source_files: list) -> bool:
        """
        Check if processed data is fresher than source files.

        Parameters:
            dataset_name (str): Name of processed dataset
            source_files (list): List of source file paths

        Returns:
            bool: True if processed data exists and is up-to-date
        """
        processed_file = self.get_processed_data_path(dataset_name)

        if not processed_file.exists():
            return False

        processed_mtime = processed_file.stat().st_mtime

        # Check if any source file is newer than processed file
        for source_file in source_files:
            source_path = Path(source_file)
            if source_path.exists() and source_path.stat().st_mtime > processed_mtime:
                return False

        return True

    def save_processed_data(self, data: Any, dataset_name: str,
                          metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save processed data with metadata.

        Parameters:
            data: Data to save (DataFrame, dict, array, etc.)
            dataset_name (str): Name for the dataset
            metadata (dict, optional): Additional metadata

        Returns:
            Path: Path to saved file
        """
        file_path = self.get_processed_data_path(dataset_name)

        # Prepare data package
        data_package = {
            'data': data,
            'metadata': metadata or {},
            'processing_timestamp': pd.Timestamp.now().isoformat(),
        }

        # Save based on data type
        if isinstance(data, pd.DataFrame):
            data.to_pickle(file_path)
        elif isinstance(data, (dict, list)):
            with open(file_path.with_suffix('.json'), 'w') as f:
                json.dump(data_package, f, indent=2, default=str)
        elif isinstance(data, np.ndarray):
            np.save(file_path.with_suffix('.npy'), data)
            # Save metadata separately
            metadata_file = file_path.with_suffix('.meta.json')
            with open(metadata_file, 'w') as f:
                json.dump(data_package['metadata'], f, indent=2, default=str)
        else:
            # Fallback: try pickle
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data_package, f)

        return file_path

    def load_processed_data(self, dataset_name: str) -> Optional[Any]:
        """
        Load processed data if it exists.

        Parameters:
            dataset_name (str): Name of the dataset

        Returns:
            Data or None if not found
        """
        file_path = self.get_processed_data_path(dataset_name)

        if not file_path.exists():
            return None

        try:
            if file_path.suffix == '.pkl':
                return pd.read_pickle(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    package = json.load(f)
                return package.get('data')
            elif file_path.suffix == '.npy':
                return np.load(file_path)
            else:
                import pickle
                with open(file_path, 'rb') as f:
                    package = pickle.load(f)
                return package.get('data')
        except Exception:
            return None

    def compute_data_hash(self, data: Any) -> str:
        """
        Compute hash of data for change detection.

        Parameters:
            data: Data to hash

        Returns:
            str: SHA256 hash of the data
        """
        if isinstance(data, pd.DataFrame):
            data_str = data.to_csv().encode()
        elif isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str).encode()
        elif isinstance(data, np.ndarray):
            data_str = data.tobytes()
        else:
            data_str = str(data).encode()

        return hashlib.sha256(data_str).hexdigest()

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """
        Process the data. Must be implemented by subclasses.

        Returns:
            Processed data
        """
        pass

    def validate_data(self, data: Any) -> bool:
        """
        Validate processed data. Can be overridden by subclasses.

        Parameters:
            data: Data to validate

        Returns:
            bool: True if data is valid
        """
        return data is not None and len(data) > 0 if hasattr(data, '__len__') else data is not None
