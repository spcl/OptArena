# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""MPI data-distribution descriptors -- the single source of truth for how a global
array is partitioned across an MPI processor grid.

Both the C-driver codegen (``bindings/mpi_driver.py``) and the mpi4py launcher
(``mpi_py_driver.py``) consume this module, so scatter and gather can never disagree
-- the property the whole MPI track's correctness rests on. Everything here is pure
numpy (no ``mpi4py``), so the exhaustive distribution tests run in ordinary CI with no
cluster.

A distribution assigns every element of a global array to exactly one rank (or to all
ranks, for ``replicated``). Per array axis the scheme is one of:

* ``block``        -- one contiguous, load-balanced block per grid coordinate (the
                      structured-stencil choice; supports a ``halo`` ghost margin).
* ``block_cyclic`` -- ScaLAPACK-style: tiles of ``tile`` elements dealt round-robin
                      across the grid coordinate (``owner(i) = (i // tile) % P``).
* ``cyclic``       -- ``block_cyclic`` with ``tile == 1``.
* ``replicated``   -- the whole axis on every rank.

Ranks and grid coordinates map row-major (``rank = coords . strides``). A grid dim of
``P`` splits the axis it owns into ``P`` shares; array axes not bound to a grid dim are
replicated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

SCHEMES = ("block", "block_cyclic", "cyclic", "replicated")


@dataclass(frozen=True)
class AxisDist:
    """How ONE array axis is laid out across the grid.

    ``grid_dim is None`` => the axis is replicated (full extent on every rank). Otherwise
    the axis is split across ``grid.dims[grid_dim]`` coordinates by ``scheme`` (``tile``
    is the block width for ``block_cyclic``; ignored by ``block``/``cyclic``). ``halo`` is
    the ghost margin added on each side of a ``block`` share (clamped at the global edge).
    """
    grid_dim: Optional[int] = None
    scheme: str = "block"
    tile: int = 1
    halo: int = 0


@dataclass(frozen=True)
class ArrayDist:
    """The distribution of one logical array: one :class:`AxisDist` per array dimension.

    ``replicated=True`` is the shorthand for "the whole array on every rank" (no axis is
    split); ``axes`` is then ignored.
    """
    axes: Tuple[AxisDist, ...] = ()
    replicated: bool = False


@dataclass(frozen=True)
class Grid:
    """The processor grid. ``math.prod(dims)`` must equal the rank count."""
    dims: Tuple[int, ...]

    @property
    def nranks(self) -> int:
        return math.prod(self.dims) if self.dims else 1

    def coords_of(self, rank: int) -> Tuple[int, ...]:
        """Row-major rank -> grid coordinates."""
        return tuple((rank // stride) % self.dims[i] for i, stride in enumerate(self._strides()))

    def rank_of(self, coords: Sequence[int]) -> int:
        return int(sum(c * s for c, s in zip(coords, self._strides())))

    def _strides(self) -> List[int]:
        strides = [1] * len(self.dims)
        for i in range(len(self.dims) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.dims[i + 1]
        return strides


def _block_bounds(n: int, parts: int, coord: int) -> Tuple[int, int]:
    """Load-balanced contiguous block ``[lo, hi)`` of ``range(n)`` for ``coord`` of
    ``parts`` (the first ``n % parts`` coordinates get one extra element)."""
    base, rem = divmod(n, parts)
    lo = coord * base + min(coord, rem)
    hi = lo + base + (1 if coord < rem else 0)
    return lo, hi


def owned_indices(n: int, axis: AxisDist, grid: Grid, coords: Sequence[int]) -> np.ndarray:
    """The global indices of a length-``n`` axis owned by grid ``coords`` under ``axis``.

    Excludes the halo (that is the owned *interior*); use :func:`halo_slice` for the
    ghost-padded read extent. For a replicated axis this is ``arange(n)`` on every rank.
    """
    if axis.grid_dim is None:
        return np.arange(n, dtype=np.int64)
    parts = grid.dims[axis.grid_dim]
    coord = coords[axis.grid_dim]
    if axis.scheme == "block":
        lo, hi = _block_bounds(n, parts, coord)
        return np.arange(lo, hi, dtype=np.int64)
    if axis.scheme in ("block_cyclic", "cyclic"):
        tile = 1 if axis.scheme == "cyclic" else max(1, axis.tile)
        idx = np.arange(n, dtype=np.int64)
        return idx[(idx // tile) % parts == coord]
    raise ValueError(f"unknown scheme {axis.scheme!r}; known: {SCHEMES}")


def halo_slice(n: int, axis: AxisDist, grid: Grid, coords: Sequence[int]) -> Tuple[int, int]:
    """The ghost-padded read extent ``[lo, hi)`` for a ``block`` axis with ``halo`` (the
    owned interior widened by ``halo`` on each side, clamped to ``[0, n]``). Only defined
    for contiguous ``block``; other schemes have no contiguous halo."""
    if axis.scheme != "block" or axis.grid_dim is None:
        raise ValueError("halo_slice is only defined for a distributed 'block' axis")
    lo, hi = _block_bounds(n, grid.dims[axis.grid_dim], coords[axis.grid_dim])
    if hi <= lo:  # this rank owns no interior on this axis -> it reads no ghost margin
        return lo, lo
    return max(0, lo - axis.halo), min(n, hi + axis.halo)


def _axis_index_lists(shape: Sequence[int], dist: ArrayDist, grid: Grid, coords: Sequence[int]) -> List[np.ndarray]:
    if len(dist.axes) != len(shape):
        raise ValueError(f"ArrayDist has {len(dist.axes)} axes but the array has {len(shape)} dimension(s)")
    return [owned_indices(shape[d], dist.axes[d], grid, coords) for d in range(len(shape))]


def local_shape(shape: Sequence[int], dist: ArrayDist, grid: Grid, rank: int) -> Tuple[int, ...]:
    """The shape of ``rank``'s local (owned-interior) tile."""
    if dist.replicated:
        return tuple(shape)
    coords = grid.coords_of(rank)
    return tuple(len(ix) for ix in _axis_index_lists(shape, dist, grid, coords))


def scatter(a: np.ndarray, dist: ArrayDist, grid: Grid) -> List[np.ndarray]:
    """Partition ``a`` into one contiguous local tile per rank (indexed by linear rank).

    The reference the drivers' ``MPI_Scatterv``/mpi4py ``Scatterv`` must reproduce. A
    ``replicated`` array yields a full copy per rank. ``halo`` is NOT applied here (it is
    a read-margin the driver adds); this returns owned interiors, so
    :func:`gather` inverts it exactly.
    """
    if dist.replicated:
        return [a.copy() for _ in range(grid.nranks)]
    tiles: List[np.ndarray] = []
    for rank in range(grid.nranks):
        coords = grid.coords_of(rank)
        tiles.append(a[np.ix_(*_axis_index_lists(a.shape, dist, grid, coords))].copy())
    return tiles


def gather(tiles: Sequence[np.ndarray], dist: ArrayDist, grid: Grid, global_shape: Sequence[int],
           dtype: np.dtype) -> np.ndarray:
    """Reconstruct the global array from per-rank owned-interior tiles -- the exact
    inverse of :func:`scatter`. For ``replicated``, rank 0's copy is authoritative."""
    out = np.empty(tuple(global_shape), dtype=dtype)
    if dist.replicated:
        out[...] = tiles[0]
        return out
    for rank in range(grid.nranks):
        coords = grid.coords_of(rank)
        out[np.ix_(*_axis_index_lists(global_shape, dist, grid, coords))] = tiles[rank]
    return out


def is_partition(shape: Sequence[int], dist: ArrayDist, grid: Grid) -> bool:
    """``True`` iff the owned interiors tile the global array exactly once (disjoint +
    complete). ``replicated`` is a partition by convention (rank 0 owns). The
    completeness invariant the roundtrip rests on -- asserted directly in tests."""
    if dist.replicated:
        return True
    seen = np.zeros(tuple(shape), dtype=np.int64)
    for rank in range(grid.nranks):
        coords = grid.coords_of(rank)
        seen[np.ix_(*_axis_index_lists(shape, dist, grid, coords))] += 1
    return bool(np.all(seen == 1))


def factor_grid(nranks: int, ndim: int) -> Grid:
    """A near-square ``ndim``-dimensional grid whose dims multiply to ``nranks`` (the
    default processor grid when the agent does not name one). Trailing dims absorb the
    remaining factors so the product is exact."""
    dims = [1] * max(1, ndim)
    remaining = nranks
    for i in range(len(dims)):
        if remaining == 1:
            break
        root = round(remaining**(1.0 / (len(dims) - i)))
        d = next((c for c in range(max(1, root), 0, -1) if remaining % c == 0), 1)
        dims[i] = d
        remaining //= d
    dims[-1] *= remaining
    return Grid(tuple(dims))


def default_distribution(shape: Sequence[int], grid: Grid, tile: int = 1) -> ArrayDist:
    """The default N-D **block-cyclic** layout: each array axis is dealt in ``tile``-wide
    blocks round-robin across the matching grid dimension (ScaLAPACK-style). Array axes
    beyond the grid rank are replicated; a size-1 grid dim leaves its axis whole."""
    # Every split (size>1) grid dim must map to an array axis, else its extra coordinates
    # would replicate -- silently double-owning the array. Require a grid whose split dims
    # fit within ndim (e.g. factor_grid(nranks, len(shape))).
    for gd in range(len(shape), len(grid.dims)):
        if grid.dims[gd] > 1:
            raise ValueError(f"grid {grid.dims} splits dimension {gd} beyond the array's "
                             f"{len(shape)} axes; use a grid with <= {len(shape)} split dims "
                             f"(e.g. factor_grid(nranks, {len(shape)}))")
    axes: List[AxisDist] = []
    for d in range(len(shape)):
        if d < len(grid.dims) and grid.dims[d] > 1:
            axes.append(AxisDist(grid_dim=d, scheme="block_cyclic", tile=max(1, tile)))
        else:
            axes.append(AxisDist(grid_dim=None))
    return ArrayDist(axes=tuple(axes))
