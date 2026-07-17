#!/usr/bin/env python
import os

from setuptools import find_packages, setup

# The numpyto_* translators live under optarena/numpy_translators/src and ship as part
# of THIS distribution (not a separate project). Their top-level import names
# (numpyto_c, numpyto_common, ...) are kept; package_dir maps each to its src location.
_TRANSLATOR_SRC = 'optarena/numpy_translators/src'
_translator_packages = find_packages(where=_TRANSLATOR_SRC)
_translator_top = [p for p in _translator_packages if '.' not in p]

setup(
    name='optarena',
    version='0.1',
    url='https://github.com/spcl/OptArena',
    author='SPCL @ ETH Zurich',
    author_email='yakupkoray.budanaz@inf.ethz.ch',
    description='OptArena',
    license='GPL-3.0-or-later',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    packages=(find_packages(include=['optarena', 'optarena.*'], exclude=['optarena.numpy_translators*']) +
              _translator_packages),
    package_dir={
        '': '.',
        **{
            p: os.path.join(_TRANSLATOR_SRC, p)
            for p in _translator_top
        }
    },
    # Ship the library's data files. include_package_data alone is not enough for a
    # `pip wheel .` build: without an active VCS file-finder it silently drops these
    # (config.yaml is loaded at import by config.py; the schema/taxonomy/env specs by
    # spec.py / languages.py). Explicit package_data globs are deterministic per-env.
    include_package_data=True,
    package_data={
        'optarena': [
            'config.yaml',
            'container_backends.txt',
            'envs/*.yaml',
            # A build input, not data: CPU_BASELINE_GCC -include's it on every gcc/g++
            # compile, so without it native C/C++ kernels do not compile from a wheel.
            'envs/vecmath.h',
        ],
    },
    # What the LIBRARY itself needs to import -- every module-level third-party import under
    # optarena/ outside benchmarks/ and tests/. requirements/<hw>.txt stays the FRAMEWORK MATRIX
    # (torch/jax/tvm/triton/numba/pythran/xgboost + the hdf5/netcdf bindings): those are
    # hardware-dependent and are chosen per platform, which is why they are not pinned here.
    #
    # Without this, `pip install optarena` (or `pip install -e .`) yields an unimportable package:
    # it resolves ZERO dependencies. That was survivable only because every in-repo consumer
    # installs `-r requirements/<hw>.txt` FIRST (.github/actions/setup, containers/*.def,
    # optarena.Dockerfile) -- a downstream repo doing just `pip install -e .` gets
    # ModuleNotFoundError one dep at a time (sqlmodel, via optarena/frameworks/schema.py, is how
    # this surfaced).
    #
    # dace is deliberately ABSENT: the PyPI wheel is an old release that imports the numpy-2-removed
    # ``np.int``. Consumers install spcl/dace@extended editable instead (see requirements/cpu.txt).
    install_requires=[
        'numpy>=2,<3',  # the array type of every benchmark + translator signature
        'scipy',  # optarena/support/helpers/sparse/generators.py, plotting
        'pandas',  # optarena/plotting.py
        'matplotlib',  # optarena/plotting.py
        'ml_dtypes',  # optarena/precision.py -- the low-precision dtypes
        'pyyaml',  # optarena/config.py + languages.py, read at import
        'sqlmodel',  # optarena/frameworks/schema.py -- the typed results-DB schema
        'jinja2',  # optarena/harness/prompts.py
        'cffi',  # optarena/harness/native_call.py
        'sympy',  # numpyto_common/lowering.py -- symbolic shape lowering
    ],
    entry_points={
        'console_scripts': [
            # The main CLI (serve / run / agent / ...). Without this the documented
            # `optarena <subcommand>` is unreachable after `pip install`.
            'optarena=optarena.cli:main',
            'optarena-install-apptainer=optarena.containers:install_apptainer_main',
            # The numpyto_* translator CLIs (folded in from the former standalone dist).
            'numpyto=numpyto_common.cli:main',
            'numpyto_c=numpyto_c.cli:main',
            'numpyto_fortran=numpyto_fortran.cli:main',
            'numpyto_cupy=numpyto_cupy.cli:main',
            'numpyto_numba=numpyto_numba.cli:main',
            'numpyto_pythran=numpyto_pythran.cli:main',
        ]
    },
    python_requires='>=3.10')
