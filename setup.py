#!/usr/bin/env python
from setuptools import find_packages, setup

# The numpyto_* translators are a SEPARATE distribution (optarena/NumpyTranslators
# has its own pyproject); install it alongside: `pip install ./optarena/NumpyTranslators`.
setup(name='optarena',
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
      packages=find_packages(include=['optarena', 'optarena.*'], exclude=['optarena.NumpyTranslators*']),
      include_package_data=True,
      entry_points={'console_scripts': [
          'optarena-install-apptainer=optarena.containers:install_apptainer_main',
      ]},
      python_requires='>=3.10')
