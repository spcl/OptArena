# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""fp8 (E4M3 / E5M2) NATIVE EMISSION: the C / C++ / Fortran backends.

``test_fp8.py`` covers the framework side (the ``Precision`` enum, the ml_dtypes
mapping, the data generators, validation through jax). This is the other half:
the numpy->native translator actually EMITTING fp8, and the emitted source
compiling and computing the right answer.

The model -- one mechanism for all three languages
--------------------------------------------------
No language has a native fp8 scalar, so fp8 is 1-byte STORAGE with a registry
``compute`` of float32 (``numpyto_common.dtypes``):

* every READ of an fp8 element promotes it to float,
* every float op ROUNDS its result back to the fp8 grid (staying in float),
* every WRITE demotes back to the byte.

Nothing here needs operator overloading, which is why C -- which has none -- is
supported by the same mechanism as C++ and Fortran, and why no emulated fp8
*class* (C++) or derived type (Fortran) is needed at all.

The per-op round is LOAD-BEARING, not belt-and-braces. ml_dtypes rounds back to
fp8 after every op, so ``y + alpha * x`` rounds TWICE. Promoting on load and
demoting only on store would compute the whole chain in float and round once;
that is a different numerical model, and on this very kernel it puts ~10% of
elements on a different fp8 code -- values like 1.5 vs 1.75 and -8.0 vs -9.0,
which no honest "fp8-appropriate tolerance" covers (they exceed even the fp8_e4m3
band of rtol=1e-1 in ``frameworks.test.TOLERANCES``). With the per-op round the
emitted kernel is BIT-EXACT against the ml_dtypes oracle, so these tests assert
exact equality rather than a tolerance: an fp8 tolerance here would only be a
place for a real bug to hide. (For reference, the tolerance the format alone
would justify: E4M3 keeps 3 mantissa bits -> eps = 2^-3 = 0.125, half-ulp
relative error 6.25e-2; E5M2 keeps 2 -> eps = 0.25, half-ulp 1.25e-1.)
"""
import ctypes
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numerical_oracle as no  # noqa: E402

ml_dtypes = pytest.importorskip("ml_dtypes")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "optarena" / "numpy_translators" / "src"))
from numpyto_common import dtypes  # noqa: E402

#: The two OCP fp8 formats: (CLI ``--precision`` spelling, canonical registry dtype,
#: ml_dtypes type). The CLI takes the enum spelling, which the registry aliases.
FP8_FORMATS = [
    ("fp8_e4m3", "float8_e4m3", "float8_e4m3fn"),
    ("fp8_e5m2", "float8_e5m2", "float8_e5m2"),
]

#: The elementwise reference kernel: ``y[i] = y[i] + alpha * x[i]``. Two ops, so
#: it is exactly the case that distinguishes per-op rounding from round-on-store.
KERNEL = "scaled_add"

#: backend -> the toolchain binary it needs (each test gates on its own).
_TOOL = {"c": "gcc", "cpp": "g++", "fortran": "gfortran"}
_EXT = {"c": ".c", "cpp": ".cpp", "fortran": ".f90"}


def _emit_fp8(tmp_path, precision):
    """Emit ``KERNEL`` at ``precision`` into ``tmp_path`` via the same CLI path
    the numerical oracle uses. Returns the output dir."""
    from optarena.emit_bridge import legacy_bench_info_dict
    from optarena.spec import BenchSpec
    info = legacy_bench_info_dict(BenchSpec.load(KERNEL))["benchmark"]
    assert no._emit(KERNEL, info, tmp_path, precision=precision), f"{KERNEL}: fp8 emit failed at {precision}"
    return tmp_path


def _src(tmp_path, precision, backend):
    """The single emitted source for ``backend`` at ``precision``."""
    hits = sorted(tmp_path.glob(f"{KERNEL}_{precision}{_EXT[backend]}"))
    assert len(hits) == 1, f"expected one {backend} source for {precision}, got {hits}"
    return hits[0]


# --------------------------------------------------------------------------- #
# 1. The registry resolves both fp8 formats to a 1-byte C / Fortran type        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cli,canon,mlname", FP8_FORMATS)
def test_registry_resolves_fp8_to_one_byte(cli, canon, mlname):
    """Both formats resolve -- through every spelling -- to a 1-byte storage type
    in C AND Fortran, and to the matching ml_dtypes type on the numpy side."""
    info = dtypes.info(cli)
    assert dtypes.canonical(cli) == canon
    assert dtypes.canonical(canon) == canon
    # A distinct C typedef, NOT a bare uint8_t: uint8_t would match the emitter's
    # narrow-int predicate and get silently widened to int64 on read.
    assert info.c == f"__npb_fp8_{canon.removeprefix('float8_')}"
    assert "uint8_t" not in info.c
    # Fortran HAS a 1-byte interop integer, so fp8 is expressible there (unlike
    # float16, whose row is fortran=None -- there is no C-interop 16-bit real).
    assert info.fortran == "integer(c_int8_t)"
    assert dtypes.fortran_kind(cli) == "integer(c_int8_t)"
    # 1 byte on the marshalling side, matching the ml_dtypes itemsize.
    assert ctypes.sizeof(dtypes.ctype_for(cli)) == 1
    assert np.dtype(getattr(ml_dtypes, mlname)).itemsize == 1
    # Storage-only: arithmetic is done in float32, not in the byte.
    assert dtypes.is_storage_only(cli) is True
    assert dtypes.compute_dtype(cli) == "float32"


def test_fp8_registry_does_not_disturb_other_dtypes():
    """The storage/compute split is fp8-only -- every other dtype computes in
    itself, so the fp32 / fp64 emission paths are unaffected."""
    for dt in ("float64", "float32", "float16", "int32", "int8", "uint8", "bool"):
        assert dtypes.is_storage_only(dt) is False, f"{dt} wrongly marked storage-only"
        assert dtypes.compute_dtype(dt) == dtypes.canonical(dt)
    # fp8 must not collide with int8 despite sharing the Fortran storage type.
    assert dtypes.fortran_kind("int8") == dtypes.fortran_kind("fp8_e4m3")
    assert dtypes.canonical("int8") != dtypes.canonical("fp8_e4m3")
    assert dtypes.c_type("int8") != dtypes.c_type("fp8_e4m3")


# --------------------------------------------------------------------------- #
# 2 + 3. It EMITS for c / cpp / fortran, and the emitted source COMPILES        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["c", "cpp", "fortran"])
@pytest.mark.parametrize("cli,canon,mlname", FP8_FORMATS)
def test_fp8_emits_and_compiles(tmp_path, cli, canon, mlname, backend):
    """``--precision fp8_*`` emits a source whose element type is the 1-byte fp8
    storage type, and that source compiles."""
    if shutil.which(_TOOL[backend]) is None:
        pytest.skip(f"{_TOOL[backend]} not installed")
    _emit_fp8(tmp_path, cli)
    src = _src(tmp_path, cli, backend)
    text = src.read_text()

    # The signature carries the storage type -- not a silently-promoted float.
    sig = text.split(f"{KERNEL}_{cli}(", 1)[1].split(")", 1)[0]
    storage = dtypes.c_type(cli) if backend != "fortran" else "integer(c_int8_t)"
    if backend == "fortran":
        assert storage in text  # Fortran declares params in the spec part
    else:
        assert storage in sig, f"{backend}: fp8 storage type missing from signature: {sig}"
        assert "double" not in sig, f"{backend}: fp8 signature leaked a double: {sig}"

    # The promote / round / demote triple is present -- i.e. the arithmetic is
    # actually being done in float and re-rounded, not on the raw bytes.
    suffix = canon.removeprefix("float8_")
    pre = "__npb_" if backend != "fortran" else "npb_"
    for fn in (f"{pre}{suffix}_to_f32", f"{pre}f32_to_{suffix}", f"{pre}rn_{suffix}"):
        assert fn in text, f"{backend}: fp8 helper {fn} not emitted"

    r = subprocess.run(no.COMPILE[backend] + [str(src), "-o", str(tmp_path / f"o_{backend}.so")],
                       capture_output=True,
                       text=True)
    assert r.returncode == 0, f"{KERNEL} {backend} {cli} compile failed:\n{r.stderr[:1500]}"


@pytest.mark.parametrize("cli,canon,mlname", FP8_FORMATS)
def test_fp8_prelude_only_when_used(tmp_path, cli, canon, mlname):
    """The fp8 prelude is injected ONLY into an fp8 kernel -- an fp64 emit is not
    carrying dead conversion helpers."""
    _emit_fp8(tmp_path / "f8", cli)
    _emit_fp8(tmp_path / "f64", "")
    suffix = canon.removeprefix("float8_")
    assert f"__npb_rn_{suffix}" in (tmp_path / "f8" / f"{KERNEL}_{cli}.c").read_text()
    fp64_c = sorted((tmp_path / "f64").glob(f"{KERNEL}_fp64.c"))[0].read_text()
    assert "__npb_rn_" not in fp64_c, "fp8 helpers leaked into the fp64 emit"
    assert "__npb_fp8" not in fp64_c


# --------------------------------------------------------------------------- #
# 4. Numeric: the compiled kernel vs the ml_dtypes fp8 reference                #
# --------------------------------------------------------------------------- #


def _run_scaled_add(so, symbol, x8, y8, alpha8):
    """Call the compiled fp8 scaled_add over raw 1-byte storage; return the
    mutated ``y`` bytes. fp8 arrays marshal as uint8 -- the ABI is the byte."""
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, symbol)
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_uint8]
    xb = np.ascontiguousarray(x8).view(np.uint8).copy()
    yb = np.ascontiguousarray(y8).view(np.uint8).copy()
    fn(xb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), yb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
       ctypes.c_int64(xb.size), ctypes.c_uint8(int(np.asarray(alpha8).view(np.uint8))))
    return yb


@pytest.mark.parametrize("backend", ["c", "cpp", "fortran"])
@pytest.mark.parametrize("cli,canon,mlname", FP8_FORMATS)
def test_fp8_numeric_matches_numpy_oracle(tmp_path, cli, canon, mlname, backend):
    """The compiled fp8 kernel reproduces the numpy ml_dtypes reference EXACTLY.

    The oracle is the kernel's own numpy reference (``y[i] + alpha * x[i]``)
    evaluated at fp8: ml_dtypes rounds each op back to fp8, so this is a
    two-rounding expression. Asserting bit-equality (rather than an fp8-width
    tolerance) is deliberate -- see the module docstring: the emitted code
    implements the same per-op rounding, so any difference at all is a real
    lowering bug, and a tolerance would only hide it.
    """
    if shutil.which(_TOOL[backend]) is None:
        pytest.skip(f"{_TOOL[backend]} not installed")
    f8 = getattr(ml_dtypes, mlname)
    _emit_fp8(tmp_path, cli)
    src = _src(tmp_path, cli, backend)
    so = tmp_path / f"num_{backend}.so"
    r = subprocess.run(no.COMPILE[backend] + [str(src), "-o", str(so)], capture_output=True, text=True)
    assert r.returncode == 0, f"{backend} {cli} compile failed:\n{r.stderr[:1500]}"

    # Values well inside the format's finite range (precision.safe_max) so the
    # test measures ROUNDING, not overflow-to-NaN/Inf saturation.
    rng = np.random.default_rng(0)
    n = 4096
    x = rng.uniform(-4, 4, n).astype(f8)
    y = rng.uniform(-4, 4, n).astype(f8)
    alpha = f8(1.5)

    want = y + alpha * x  # the numpy oracle: rounds to fp8 after EACH op
    got = _run_scaled_add(so, f"{KERNEL}_{cli}", x, y, alpha).view(f8)

    exact = got.view(np.uint8) == np.ascontiguousarray(want).view(np.uint8)
    if not exact.all():
        bad = np.where(~exact)[0][:5]
        detail = [(float(got.astype(np.float32)[i]), float(want.astype(np.float32)[i])) for i in bad]
        pytest.fail(f"{backend} {cli}: {(~exact).sum()}/{n} elements differ from the ml_dtypes "
                    f"oracle (got, want): {detail}")


@pytest.mark.parametrize("cli,canon,mlname", FP8_FORMATS)
def test_fp8_conversions_cover_every_code(tmp_path, cli, canon, mlname):
    """Drive all 256 fp8 codes -- including subnormals, both zeros, Inf and NaN --
    through the emitted promote / demote and match ml_dtypes on every one.

    Guards the conversion maths at the edges of the format, which the random
    numeric test above never reaches. ``y + alpha*x`` with ``alpha = 0`` puts
    exactly one promote + one demote on each code.

    Note this is NOT asserted to be an identity on the code: ``-0.0 + 0.0`` is
    ``+0.0`` in IEEE-754, so code 0x80 legitimately becomes 0x00. numpy does the
    same, which is precisely why the ORACLE is the reference here rather than the
    input -- an identity assertion would flag correct signed-zero behaviour.
    """
    if shutil.which("gcc") is None:
        pytest.skip("gcc not installed")
    f8 = getattr(ml_dtypes, mlname)
    _emit_fp8(tmp_path, cli)
    so = tmp_path / "rt.so"
    r = subprocess.run(no.COMPILE["c"] + [str(_src(tmp_path, cli, "c")), "-o", str(so)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:1500]

    vals = np.arange(256, dtype=np.uint8).view(f8)
    zeros = np.zeros(256, dtype=f8)
    want = vals + f8(0.0) * zeros
    got = _run_scaled_add(so, f"{KERNEL}_{cli}", zeros, vals, f8(0.0))

    bad = np.where(got != np.ascontiguousarray(want).view(np.uint8))[0]
    assert not len(bad), (f"{cli}: codes {[hex(int(b)) for b in bad]} disagree with ml_dtypes "
                          f"through the emitted promote/demote")
