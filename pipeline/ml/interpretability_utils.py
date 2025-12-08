"""
Interpretability Utilities
==========================

Helpers to map latent indices back to coarse modality/feature blocks and to
summarize feature importance in physically meaningful labels.
"""

from typing import List, Dict, Any


def map_latent_to_feature_block(latent_index: int, modality_blocks: List[Dict[str, Any]]) -> str:
    """
    Map a latent dimension index to a coarse modality/feature block label.

    Parameters
    ----------
    latent_index : int
        Index in the latent vector (e.g., 0..511).
    modality_blocks : list of dict
        Ordered list describing how latent dimensions are allocated.
        Each dict may include:
            - name: modality label (e.g., 'cmb_planck_tt_lowell')
            - start: start index (inclusive)
            - end: end index (exclusive)

    Returns
    -------
    str
        Human-readable label for the latent block; returns "unknown" if unmapped.
    """
    for block in modality_blocks:
        start = block.get("start")
        end = block.get("end")
        if start is None or end is None:
            continue
        if start <= latent_index < end:
            return block.get("name", "unknown")
    return "unknown"


def default_modality_blocks(latent_dim: int, modalities: List[str]) -> List[Dict[str, Any]]:
    """
    Construct a coarse block map by evenly partitioning the latent space
    across the observed modalities. This is a fallback when detailed encoder
    layouts are unavailable.
    """
    if not modalities:
        return [{"name": "unknown", "start": 0, "end": latent_dim}]

    block_size = max(1, latent_dim // len(modalities))
    blocks = []
    cursor = 0
    for name in modalities:
        start = cursor
        end = min(latent_dim, start + block_size)
        blocks.append({"name": name, "start": start, "end": end})
        cursor = end
    # Ensure coverage of any trailing dims
    if cursor < latent_dim:
        blocks.append({"name": "residual", "start": cursor, "end": latent_dim})
    return blocks

