"""
Catalog Registry
================

Registry for galaxy catalog providers to enable extensibility.
"""

from typing import Dict, Type
from .base_catalog import BaseGalaxyCatalog
from .sdss_dr16_catalog import SDSSDR16Catalog
from .sdss_dr7_catalog import SDSSDR7Catalog


class CatalogRegistry:
    """
    Registry for galaxy catalog providers.
    
    Enables easy addition of new catalog types.
    """
    
    _catalogs: Dict[str, Type[BaseGalaxyCatalog]] = {
        'sdss_dr16': SDSSDR16Catalog,
        'sdss_dr7': SDSSDR7Catalog,
    }
    
    @classmethod
    def register(cls, name: str, catalog_class: Type[BaseGalaxyCatalog]):
        """
        Register a new catalog provider.
        
        Parameters:
            name: Catalog identifier name
            catalog_class: Catalog class (subclass of BaseGalaxyCatalog)
        """
        if not issubclass(catalog_class, BaseGalaxyCatalog):
            raise TypeError(f"Catalog class must subclass BaseGalaxyCatalog")
        cls._catalogs[name] = catalog_class
    
    @classmethod
    def get(cls, name: str, downloaded_data_dir, processed_data_dir) -> BaseGalaxyCatalog:
        """
        Get catalog provider instance.
        
        Parameters:
            name: Catalog identifier name
            downloaded_data_dir: Directory for downloaded data
            processed_data_dir: Directory for processed data
            
        Returns:
            Catalog provider instance
            
        Raises:
            ValueError: If catalog name not found
        """
        if name not in cls._catalogs:
            available = ', '.join(cls._catalogs.keys())
            raise ValueError(f"Unknown catalog '{name}'. Available: {available}")
        
        catalog_class = cls._catalogs[name]
        return catalog_class(downloaded_data_dir, processed_data_dir)
    
    @classmethod
    def list_available(cls) -> list:
        """Return list of available catalog names."""
        return list(cls._catalogs.keys())

