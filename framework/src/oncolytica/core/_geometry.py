"""
core/geometry.py
================
Pure functions for working with Morton codes (Z-order curve) and voxel grid
key calculations. Independent of both GPU and CPU backends.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Morton (Z-order) interleaving
# ---------------------------------------------------------------------------

def interleave_9bit(x: int) -> int:
    """Interleave the lower 9 bits of an integer for Morton code encoding."""
    x = x & 0x1FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x <<  8)) & 0x0300F00F
    x = (x | (x <<  4)) & 0x030C30C3
    x = (x | (x <<  2)) & 0x09249249
    return x


def get_chemical_voxel_key(cx: int, cy: int, cz: int) -> int:
    """Morton key for chemical voxel (coarse grid)."""
    return (
        interleave_9bit(cx)
        | (interleave_9bit(cy) << 1)
        | (interleave_9bit(cz) << 2)
    )


def get_sub_voxel_index(sx: int, sy: int, sz: int) -> int:
    """Sub-voxel index within a chemical voxel (range 0-7)."""
    return (sx & 1) | ((sy & 1) << 1) | ((sz & 1) << 2)


def get_tissue_voxel_key(sx: int, sy: int, sz: int) -> int:
    """Morton key for tissue voxel (fine grid)."""
    j = get_chemical_voxel_key(sx // 2, sy // 2, sz // 2)
    i = get_sub_voxel_index(sx, sy, sz)
    return (j << 3) | i


def voxel_table_size(tissue_grid_dim: tuple[int, int, int]) -> int:
    """Calculate the minimum required VoxelTable size for the given tissue grid."""
    t = max(tissue_grid_dim)
    return get_tissue_voxel_key(t, t, t) + 1