"""
Galaxy Catalog Providers
========================

Abstract interface and implementations for galaxy catalog providers.
"""

from .base_catalog import BaseGalaxyCatalog
from .sdss_base_catalog import SDSSBaseCatalog
from .sdss_dr16_catalog import SDSSDR16Catalog
from .sdss_dr7_catalog import SDSSDR7Catalog
from .catalog_registry import CatalogRegistry

__all__ = ['BaseGalaxyCatalog', 'SDSSBaseCatalog', 'SDSSDR16Catalog', 'SDSSDR7Catalog', 'CatalogRegistry']

