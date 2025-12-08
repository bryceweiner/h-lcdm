"""
Anomaly Ontology Schema
======================

Defines a compact ontology for ML-detected anomalies and utilities to map
existing ML tests into ontology tags. The goal is to provide stable,
context-aware labels for downstream reporting and Grok prompts.
"""

from enum import Enum
from typing import Dict, Any, List, Set


class AnomalyTag(str, Enum):
    """Ontology tags for ML anomalies."""

    E8_GEOMETRY = "e8_geometry"
    NETWORK_CLUSTERING_ETA = "network_clustering_eta"
    GAMMA_QTEP = "gamma_qtep"
    CHIRAL_ASYMMETRY = "chiral_asymmetry"
    VOID_THERMODYNAMICS = "void_thermodynamics"
    SURVEY_SYSTEMATIC = "survey_systematic_candidate"
    MIXED_MULTIMODAL = "mixed_multimodal"
    BAO_SCALE_DISCREPANCY = "bao_scale_discrepancy"
    MODEL_DIFF_HLCDM = "model_diff_hlcdm"
    MODEL_DIFF_LCDM = "model_diff_lcdm"
    MODEL_DIFF_INDETERMINATE = "model_diff_indeterminate"
    
    # Data Context Tags
    CONTEXT_BAO = "context_bao"
    CONTEXT_CMB = "context_cmb"
    CONTEXT_VOIDS = "context_voids"
    CONTEXT_GALAXIES = "context_galaxies"
    CONTEXT_GW = "context_gw"
    CONTEXT_FRB = "context_frb"
    CONTEXT_LYMAN = "context_lyman"
    CONTEXT_JWST = "context_jwst"


def map_test_to_tags(test_name: str, test_result: Dict[str, Any]) -> Set[AnomalyTag]:
    """
    Map existing ML test outputs into ontology tags.

    Parameters
    ----------
    test_name : str
        Name of the ML interpretability/diagnostic test.
    test_result : dict
        Structured result for the given test.

    Returns
    -------
    Set[AnomalyTag]
        Ontology tags implied by the test outcome.
    """
    tags: Set[AnomalyTag] = set()
    if test_name == "e8_pattern":
        if test_result.get("e8_signature_detected"):
            tags.add(AnomalyTag.E8_GEOMETRY)
        # Network clustering close to eta can also imply clustering tag
        network = test_result.get("network_analysis", {})
        eta_clust = network.get("clustering_coefficient")
        if eta_clust is not None:
            tags.add(AnomalyTag.NETWORK_CLUSTERING_ETA)
    elif test_name == "network_analysis":
        tags.add(AnomalyTag.NETWORK_CLUSTERING_ETA)
    elif test_name == "chirality":
        if test_result.get("chirality_detected"):
            tags.add(AnomalyTag.CHIRAL_ASYMMETRY)
    elif test_name == "gamma_qtep":
        pattern = test_result.get("pattern_analysis", {})
        if pattern.get("qtep_consistent") or pattern.get("pattern_detected"):
            tags.add(AnomalyTag.GAMMA_QTEP)
    else:
        # Default catch-all: keep mixed multimodal if nothing specific is known
        tags.add(AnomalyTag.MIXED_MULTIMODAL)
    return tags


def model_diff_tag(favored_model: str) -> AnomalyTag:
    """
    Convert a favored model string into an ontology tag.
    """
    favored_model = (favored_model or "").upper()
    if favored_model == "HLCDM":
        return AnomalyTag.MODEL_DIFF_HLCDM
    if favored_model == "LCDM":
        return AnomalyTag.MODEL_DIFF_LCDM
    return AnomalyTag.MODEL_DIFF_INDETERMINATE


def merge_tags(*tag_sets: List[Set[AnomalyTag]]) -> List[str]:
    """
    Merge multiple tag sets and return a sorted, de-duplicated list of strings.
    """
    merged: Set[AnomalyTag] = set()
    for ts in tag_sets:
        # Filter out Nones if any passed
        if ts:
            merged.update(ts)
    return sorted({str(t) for t in merged})
