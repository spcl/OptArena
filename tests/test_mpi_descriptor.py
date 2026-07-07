# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exhaustive host-side tests for the MPI data-distribution math
(``optarena.agent_bench.mpi_descriptor``) -- the correctness core the whole MPI track
rests on. No cluster / mpi4py needed: scatter/gather are pure numpy, so the full
scheme x dimensionality x grid x ragged-size x dtype matrix runs in ordinary CI.

The two load-bearing invariants:
  * ``gather(scatter(A)) == A`` bit-exact (scatter/gather come from the SAME descriptor),
  * the owned interiors form an exact PARTITION (disjoint + complete).
"""
import math

import numpy as np
import pytest

from optarena.agent_bench.envelope import Submission
from optarena.agent_bench.mpi_descriptor import (AxisDist, ArrayDist, Descriptor, Grid, default_distribution,
                                                 factor_grid, gather, halo_slice, is_partition, local_shape,
                                                 owned_indices, scatter)
from optarena.bindings.contract import Arg, Binding

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def _arange(shape, dtype):
    return np.arange(math.prod(shape), dtype=dtype).reshape(shape)


def _dist_1d(scheme, parts, tile=1):
    return Grid((parts, )), ArrayDist(axes=(AxisDist(grid_dim=0, scheme=scheme, tile=tile), ))


# --- Grid rank <-> coords is a bijection --------------------------------------------


@pytest.mark.parametrize("dims", [(1, ), (4, ), (2, 3), (2, 2, 2), (1, 4), (4, 1), (2, 1, 3)])
def test_grid_rank_coords_roundtrip(dims):
    g = Grid(dims)
    assert g.nranks == math.prod(dims)
    seen = set()
    for r in range(g.nranks):
        c = g.coords_of(r)
        assert all(0 <= ci < di for ci, di in zip(c, dims))
        assert g.rank_of(c) == r
        seen.add(c)
    assert len(seen) == g.nranks  # every coordinate hit exactly once


# --- 1D block bounds: balanced + contiguous + complete ------------------------------


@pytest.mark.parametrize("n,parts", [(10, 3), (12, 4), (7, 4), (1, 4), (5, 5), (100, 7), (3, 8)])
def test_block_bounds_partition_and_balance(n, parts):
    g, dist = _dist_1d("block", parts)
    idxs = [owned_indices(n, dist.axes[0], g, g.coords_of(r)) for r in range(parts)]
    # contiguous
    for ix in idxs:
        if len(ix):
            assert list(ix) == list(range(ix[0], ix[-1] + 1))
    # exact cover of range(n)
    cover = np.concatenate(idxs) if any(len(i) for i in idxs) else np.array([], np.int64)
    assert sorted(cover.tolist()) == list(range(n))
    # balanced: sizes differ by at most 1
    sizes = [len(ix) for ix in idxs]
    assert max(sizes) - min(sizes) <= 1


# --- the big roundtrip matrix -------------------------------------------------------

_SHAPES_1D = [(12, ), (13, ), (1, ), (7, ), (256, )]
_SHAPES_2D = [(6, 6), (7, 5), (5, 7), (1, 9), (9, 1), (2, 2)]
_SHAPES_3D = [(4, 4, 4), (5, 3, 2), (2, 7, 1)]
_SHAPES_4D = [(2, 2, 2, 2), (3, 2, 4, 1)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheme", ["block", "block_cyclic", "cyclic", "replicated"])
@pytest.mark.parametrize("parts", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("shape", _SHAPES_1D)
def test_roundtrip_1d(shape, parts, scheme, dtype):
    a = _arange(shape, dtype)
    if scheme == "replicated":
        g, dist = Grid((parts, )), ArrayDist(replicated=True)
    else:
        g, dist = _dist_1d(scheme, parts, tile=2)
    tiles = scatter(a, dist, g)
    assert len(tiles) == parts
    assert [t.shape for t in tiles] == [local_shape(shape, dist, g, r) for r in range(parts)]
    back = gather(tiles, dist, g, shape, np.dtype(dtype))
    assert np.array_equal(back, a) and back.dtype == a.dtype
    assert is_partition(shape, dist, g)


@pytest.mark.parametrize("dtype", [np.float64, np.int32])
@pytest.mark.parametrize("grid_dims,axes", [
    ((2, 3), (("block", "block"))),
    ((2, 2), (("block_cyclic", "block"))),
    ((3, 2), (("cyclic", "cyclic"))),
    ((4, 1), (("block", "replicated_axis"))),
    ((2, 2), (("block_cyclic", "block_cyclic"))),
])
@pytest.mark.parametrize("shape", _SHAPES_2D)
def test_roundtrip_2d_grid(shape, grid_dims, axes, dtype):
    g = Grid(grid_dims)
    axdefs = []
    for d, sch in enumerate(axes):
        if sch == "replicated_axis":
            axdefs.append(AxisDist(grid_dim=None))
        else:
            axdefs.append(AxisDist(grid_dim=d, scheme=sch, tile=2))
    dist = ArrayDist(axes=tuple(axdefs))
    a = _arange(shape, dtype)
    tiles = scatter(a, dist, g)
    back = gather(tiles, dist, g, shape, np.dtype(dtype))
    assert np.array_equal(back, a)
    assert is_partition(shape, dist, g)


@pytest.mark.parametrize("scheme", ["block", "block_cyclic", "cyclic"])
@pytest.mark.parametrize("shape", _SHAPES_3D + _SHAPES_4D)
def test_roundtrip_nd_leading_axis(shape, scheme):
    # distribute only the leading axis over a 1D grid of ranks (the common stencil case)
    for parts in (1, 2, 3):
        g = Grid((parts, ) + (1, ) * (len(shape) - 1))
        axes = [AxisDist(grid_dim=0, scheme=scheme, tile=2)] + [AxisDist(grid_dim=None)] * (len(shape) - 1)
        dist = ArrayDist(axes=tuple(axes))
        a = _arange(shape, np.float64)
        back = gather(scatter(a, dist, g), dist, g, shape, np.dtype(np.float64))
        assert np.array_equal(back, a), (shape, scheme, parts)
        assert is_partition(shape, dist, g)


# --- block-cyclic ownership formula -------------------------------------------------


def test_block_cyclic_owner_formula():
    # owner(i) = (i // tile) % parts -- verify against the descriptor
    n, parts, tile = 20, 3, 2
    g, dist = _dist_1d("block_cyclic", parts, tile=tile)
    for coord in range(parts):
        got = set(owned_indices(n, dist.axes[0], g, (coord, )).tolist())
        want = {i for i in range(n) if (i // tile) % parts == coord}
        assert got == want


# --- halo (structured-stencil read margin) ------------------------------------------


@pytest.mark.parametrize("n,parts,halo", [(12, 3, 1), (10, 4, 2), (7, 3, 1), (5, 5, 1)])
def test_halo_slice_widens_interior_and_clamps(n, parts, halo):
    g = Grid((parts, ))
    ax = AxisDist(grid_dim=0, scheme="block", tile=1, halo=halo)
    for coord in range(parts):
        interior = owned_indices(n, AxisDist(grid_dim=0, scheme="block"), g, (coord, ))
        lo, hi = halo_slice(n, ax, g, (coord, ))
        assert 0 <= lo <= hi <= n
        if len(interior):
            # ghost margin present except where clamped at the global boundary
            assert lo == max(0, interior[0] - halo)
            assert hi == min(n, interior[-1] + 1 + halo)
        # neighbours' halos overlap the interior they read from
        assert hi - lo >= len(interior)


def test_halo_slice_rejects_non_block():
    g = Grid((2, ))
    with pytest.raises(ValueError):
        halo_slice(10, AxisDist(grid_dim=0, scheme="cyclic", halo=1), g, (0, ))


# --- factor_grid + default_distribution ---------------------------------------------


@pytest.mark.parametrize("nranks", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_factor_grid_product_exact(nranks, ndim):
    g = factor_grid(nranks, ndim)
    assert g.nranks == nranks
    assert len(g.dims) == ndim


@pytest.mark.parametrize("nranks", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("shape", [(12, ), (8, 8), (6, 6, 6)])
def test_default_distribution_is_a_roundtrip_partition(nranks, shape):
    g = factor_grid(nranks, len(shape))
    dist = default_distribution(shape, g, tile=2)
    a = _arange(shape, np.float64)
    back = gather(scatter(a, dist, g), dist, g, shape, np.dtype(np.float64))
    assert np.array_equal(back, a)
    assert is_partition(shape, dist, g)


# --- edge: more ranks than elements along the axis ----------------------------------


@pytest.mark.parametrize("scheme", ["block", "block_cyclic", "cyclic"])
def test_more_ranks_than_elements(scheme):
    shape, parts = (3, ), 5  # 5 ranks, 3 elements -> some ranks own nothing
    g, dist = _dist_1d(scheme, parts, tile=1)
    a = _arange(shape, np.float64)
    tiles = scatter(a, dist, g)
    assert sum(t.size for t in tiles) == a.size  # still a complete partition
    back = gather(tiles, dist, g, shape, np.dtype(np.float64))
    assert np.array_equal(back, a)


# --- negatives + edge cases the invariants themselves must catch --------------------


def test_is_partition_false_for_overlapping_dist():
    """is_partition must REJECT a bad layout, not just accept good ones (else a regression
    to always-True would pass the whole suite silently)."""
    # both array axes bound to the SAME 1-D grid dim -> ranks own only the diagonal
    # quadrants, leaving the off-diagonal ones uncovered.
    g = Grid((2, ))
    dist = ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), AxisDist(grid_dim=0, scheme="block")))
    assert is_partition((4, 4), dist, g) is False


def test_halo_slice_empty_interior_reads_no_ghost():
    """A rank that owns no interior (more ranks than elements) reads NO ghost margin."""
    n, parts, halo = 3, 5, 1  # ranks 3,4 own nothing
    g = Grid((parts, ))
    ax = AxisDist(grid_dim=0, scheme="block", halo=halo)
    for coord in (3, 4):
        lo, hi = halo_slice(n, ax, g, (coord, ))
        assert lo == hi, (coord, lo, hi)  # empty extent, not a spurious 1-cell window


def test_axis_count_mismatch_is_a_clear_error():
    """A descriptor whose axis count != array ndim fails clearly, not with an opaque
    IndexError deep inside scatter/gather."""
    with pytest.raises(ValueError, match="axes but the array"):
        scatter(np.zeros((4, )), ArrayDist(), Grid((2, )))


def test_default_distribution_replicates_trailing_axes():
    """More array axes than grid dims -> trailing axes replicate whole, still an exact
    roundtrip partition."""
    shape, g = (6, 6), Grid((2, ))  # a 1-D grid over a 2-D array
    dist = default_distribution(shape, g, tile=2)
    assert dist.axes[1].grid_dim is None  # trailing axis replicated
    a = _arange(shape, np.float64)
    back = gather(scatter(a, dist, g), dist, g, shape, np.dtype(np.float64))
    assert np.array_equal(back, a)
    assert is_partition(shape, dist, g)


def test_default_distribution_rejects_grid_wider_than_array():
    """A grid that splits more dims than the array has axes would double-own the data;
    default_distribution refuses it rather than silently producing a non-partition."""
    with pytest.raises(ValueError, match="beyond the array"):
        default_distribution((12, ), Grid((2, 2)))


# --- Descriptor.from_submission: the semantic layer over the raw distribution dict -------
#
# A synthetic 2-D kernel: input A + output C both (M, N); size symbols M, N; a value
# scalar alpha. Declarative shapes let the descriptor derive that M sizes axis 0 and N
# axis 1, so a distributed axis makes the matching symbol's LOCAL value the local extent.


def _binding_2d() -> Binding:
    args = (
        Arg(name="A", kind="ptr", dtype="float64", is_const=True, shape=("M", "N")),
        Arg(name="C", kind="ptr", dtype="float64", is_const=False, shape=("M", "N"), role="output"),
        Arg(name="M", kind="scalar", dtype="int64", is_const=True, role="symbol"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
        Arg(name="alpha", kind="scalar", dtype="float64", is_const=True),
    )
    return Binding(kernel="k2d", config="dense", args=args, symbols={})


def _sub(dist) -> Submission:
    return Submission(language="c", source="x", distribution=dist)


def _block_axis0(scheme="block", tile=1, halo=0):
    # array laid out block over grid dim 0 (the leading axis), axis 1 whole.
    return {"axes": [{"grid_dim": 0, "scheme": scheme, "tile": tile, "halo": halo}, {"grid_dim": None}]}


def test_from_submission_resolves_declared_and_replicates_the_rest():
    b = _binding_2d()
    sub = _sub({"grid": [2, 1], "arrays": {"A": _block_axis0(), "C": _block_axis0()}})
    d = Descriptor.from_submission(sub, b, ranks=2)
    assert d.grid.dims == (2, 1)
    # A and C are laid out as declared; there is no un-named array here, but a scalar is
    # never in `arrays` (scalars are broadcast by value).
    assert set(d.arrays) == {"A", "C"}
    assert d.arrays["A"].axes[0].grid_dim == 0 and d.arrays["A"].axes[1].grid_dim is None
    assert not d.arrays["A"].replicated


def test_from_submission_replicates_undeclared_array():
    b = _binding_2d()
    # only A is distributed; C is left out -> replicated (lives on every rank, rank 0 wins).
    d = Descriptor.from_submission(_sub({"grid": [2, 1], "arrays": {"A": _block_axis0()}}), b, ranks=2)
    assert d.arrays["C"].replicated is True
    assert d.arrays["A"].replicated is False


def test_from_submission_rejects_grid_rank_mismatch():
    b = _binding_2d()
    with pytest.raises(ValueError, match="configured for 4"):
        Descriptor.from_submission(_sub({"grid": [2, 1], "arrays": {"A": _block_axis0()}}), b, ranks=4)


def test_from_submission_rejects_unknown_array():
    b = _binding_2d()
    with pytest.raises(ValueError, match="unknown array 'Z'"):
        Descriptor.from_submission(_sub({"grid": [2, 1], "arrays": {"Z": _block_axis0()}}), b, ranks=2)


def test_from_submission_rejects_distributing_a_scalar():
    b = _binding_2d()
    # alpha is a scalar -> broadcast by value, never distributed.
    dist = {"grid": [2], "arrays": {"alpha": {"axes": [{"grid_dim": 0, "scheme": "block"}]}}}
    with pytest.raises(ValueError, match="scalar 'alpha'"):
        Descriptor.from_submission(_sub(dist), b, ranks=2)


def test_from_submission_rejects_axis_count_mismatch():
    b = _binding_2d()  # A is 2-D
    dist = {"grid": [2], "arrays": {"A": {"axes": [{"grid_dim": 0, "scheme": "block"}]}}}  # only 1 axis
    with pytest.raises(ValueError, match="declares 1 axis/axes but the array has 2"):
        Descriptor.from_submission(_sub(dist), b, ranks=2)


def test_from_submission_rejects_halo_on_non_block():
    b = _binding_2d()
    dist = {"grid": [2, 1], "arrays": {"A": _block_axis0(scheme="cyclic", halo=1)}}
    with pytest.raises(ValueError, match="halo=1 requires scheme 'block'"):
        Descriptor.from_submission(_sub(dist), b, ranks=2)


def test_descriptor_scatter_gather_roundtrip_declared_and_replicated():
    b = _binding_2d()
    d = Descriptor.from_submission(_sub({"grid": [2, 1], "arrays": {"A": _block_axis0()}}), b, ranks=2)
    a = _arange((8, 4), np.float64)
    # A is block-distributed on axis 0; C is replicated (undeclared). Both roundtrip.
    for name in ("A", "C"):
        back = d.gather(name, d.scatter(name, a), a.shape, np.dtype(np.float64))
        assert np.array_equal(back, a), name
    # per-rank tile shapes agree with local_shape.
    assert [t.shape for t in d.scatter("A", a)] == [d.local_shape("A", a.shape, r) for r in range(2)]


def test_descriptor_length1_array_is_replicated_and_gathers_from_rank0():
    b = _binding_2d()
    d = Descriptor.from_submission(_sub({"grid": [2, 1], "arrays": {"A": _block_axis0()}}), b, ranks=2)
    # A length-1 output (a wrapped scalar reduction) is forced replicated regardless of the
    # declared layout: every rank holds it, rank 0 is authoritative.
    scalarish = np.array([[42.0]])
    assert d.dist_for("A", scalarish.shape).replicated is True
    tiles = d.scatter("A", scalarish)
    assert all(np.array_equal(t, scalarish) for t in tiles)  # broadcast to every rank
    tiles = [np.array([[42.0]]), np.array([[999.0]])]  # ranks disagree
    back = d.gather("A", tiles, (1, 1), np.dtype(np.float64))
    assert back[0, 0] == 42.0  # rank 0 wins


def test_local_size_scalars_localises_distributed_symbol_only():
    b = _binding_2d()
    # A block over grid dim 0 (axis 0 <-> symbol M); axis 1 (symbol N) stays whole.
    d = Descriptor.from_submission(_sub({
        "grid": [2, 1],
        "arrays": {
            "A": _block_axis0(),
            "C": _block_axis0()
        }
    }),
                                   b,
                                   ranks=2)
    g = {"M": 8, "N": 4, "alpha": 2.0}
    r0, r1 = d.local_size_scalars(g, 0), d.local_size_scalars(g, 1)
    assert r0["M"] == 4 and r1["M"] == 4  # 8 split over 2 ranks
    assert r0["N"] == 4 and r0["alpha"] == 2.0  # non-distributed symbol + value scalar unchanged


def test_local_size_scalars_ragged_split():
    b = _binding_2d()
    d = Descriptor.from_submission(_sub({"grid": [3, 1], "arrays": {"A": _block_axis0()}}), b, ranks=3)
    g = {"M": 7, "N": 4}  # 7 over 3 -> 3, 2, 2
    assert [d.local_size_scalars(g, r)["M"] for r in range(3)] == [3, 2, 2]


def test_local_size_scalars_no_shapes_leaves_symbols_global():
    """A legacy binding (pointer shape unknown) has no shapes-derived symbol map, so every
    symbol stays global until an explicit `symbol_axes` override (the manifest mpi: block)."""
    args = (
        Arg(name="A", kind="ptr", dtype="float64", is_const=True),  # shape=None (legacy init)
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
    )
    b = Binding(kernel="legacy", config="dense", args=args, symbols={})
    sub = _sub({"grid": [2], "arrays": {"A": {"axes": [{"grid_dim": 0, "scheme": "block"}]}}})
    # no shapes -> global unchanged
    d = Descriptor.from_submission(sub, b, ranks=2)
    assert d.local_size_scalars({"N": 8}, 0)["N"] == 8
    # explicit manifest mapping N -> A:0 makes it local
    d2 = Descriptor.from_submission(sub, b, ranks=2, symbol_axes={"N": ("A", 0)})
    assert d2.local_size_scalars({"N": 8}, 0)["N"] == 4
