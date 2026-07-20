"""A shape-changing rebinding inside nested control flow must not leak its rename outward.

The shape-versioning pass renames a name that is re-bound to a DIFFERENT extent (``x`` ->
``x__v1``) so each extent gets its own buffer. Nested scopes were walked with the SAME rename map
rather than a copy -- the code did not do what its own comment said -- so a rename minted inside an
``if`` branch stayed active after the branch closed. Every later read of ``x`` then resolved to a
buffer that was only written on the path not taken:

    c, cpp   -> SIGSEGV (read past the end of the original allocation)
    fortran  -> silently wrong values, no diagnostic at all

The silent Fortran result is why this is pinned per backend. A rebinding that is only reachable
conditionally cannot be resolved statically at all, so the pass refuses it rather than guessing;
the unconditional top-level case keeps working exactly as before.
"""
import numpy as np
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def _run(src, n=4):
    return run_op(src, "f", {"a": np.arange(1, n + 1, dtype=np.float64)}, {"out": (n, )}, {"N": n},
                  shapes={"a": "(N,)", "out": "(N,)"}, dtypes={"a": "float64", "out": "float64"}, backends=_NATIVE)


_REBIND_IN_UNTAKEN_BRANCH = ("import numpy as np\n"
                             "def f(a, out):\n"
                             "    n = a.shape[0]\n"
                             "    x = np.zeros(n)\n"
                             "    for i in range(n):\n"
                             "        x[i] = a[i]\n"
                             "    if n > 1000:\n"
                             "        x = np.zeros(2 * n)\n"
                             "    for i in range(n):\n"
                             "        out[i] = x[i]\n")


def test_conditional_shape_rebinding_is_refused_not_miscompiled():
    # Whichever branch runs is a RUNTIME fact, so no static buffer choice is correct. Refusing is
    # the only sound answer -- silently binding to either one is the bug this pins. The oracle
    # reports an emit-time refusal as a status rather than propagating it, so assert on that.
    for backend, status in _run(_REBIND_IN_UNTAKEN_BRANCH).items():
        assert status.startswith("FAIL:emit:NotImplementedError"), f"{backend}: {status}"
        assert "conditional control flow" in status, f"{backend}: {status}"


def test_unconditional_rebinding_at_top_level_still_versions():
    # The supported case: both extents are live on every path, so each gets its own buffer.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        x[i] = a[i]\n"
           "    x = np.zeros(2 * n)\n"
           "    for i in range(2 * n):\n"
           "        x[i] = 1.0\n"
           "    for i in range(n):\n"
           "        out[i] = x[i] + a[i]\n")
    _assert_ok(_run(src))


def test_same_shape_rebinding_reuses_one_buffer():
    # A rebinding to the SAME extent is not a new buffer and must not be renamed at all, inside
    # control flow or out of it.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        x[i] = a[i]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        x[i] = a[i] * 2.0\n"
           "    for i in range(n):\n"
           "        out[i] = x[i]\n")
    _assert_ok(_run(src))


def test_sibling_loop_nests_may_reuse_a_name_at_different_shapes():
    # The shape ICON's velocity_tendencies actually has: two INDEPENDENT temporaries that happen to
    # share the name `t`, each written and fully consumed inside its own nest, at different extents.
    # Nothing reads `t` after either nest, so there is no ambiguity -- refusing this would reject a
    # working kernel, which an earlier, liveness-blind version of the guard did.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    for i in range(n):\n"
           "        t = np.zeros(n)\n"
           "        for j in range(n):\n"
           "            t[j] = a[j] * 2.0\n"
           "        out[i] = t[i]\n"
           "    for i in range(n):\n"
           "        t = np.zeros(2 * n)\n"
           "        for j in range(2 * n):\n"
           "            t[j] = a[i]\n"
           "        out[i] = out[i] + t[0]\n")
    _assert_ok(_run(src))


def test_rebinding_confined_to_a_loop_body_does_not_escape():
    # x is re-bound and fully consumed inside the loop body; nothing after the loop reads it, so
    # there is no ambiguity to refuse and the kernel must still translate.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    for i in range(n):\n"
           "        t = np.zeros(2 * n)\n"
           "        for j in range(2 * n):\n"
           "            t[j] = a[i]\n"
           "        out[i] = t[0]\n")
    _assert_ok(_run(src))
