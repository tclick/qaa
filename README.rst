Quasi-Anharmonic Analysis
=========================

|PyPI| |Python Version| |License|

|Read the Docs| |Tests| |Codecov|

|pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/qaa.svg
   :target: https://pypi.org/project/qaa/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/qaa
   :target: https://pypi.org/project/qaa
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/qaa
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License
.. |Read the Docs| image:: https://readthedocs.org/projects/pyqaa/badge/?version=latest
   :target: https://pyqaa.readthedocs.io/en/latest/?badge=latest
   :alt: ReDocumentation Status
.. |Tests| image:: https://github.com/tclick/qaa/workflows/Tests/badge.svg
   :target: https://github.com/tclick/qaa/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/tclick/qaa/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/tclick/qaa
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. image:: https://pyup.io/repos/github/tclick/qaa/shield.svg
     :target: https://pyup.io/repos/github/tclick/qaa/
     :alt: Updates
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

Features
--------

`qaa` analyzes molecular dynamics (MD) trajectories by using joint
diagonalization (JADE) to separate the information. The JADE [1]_ and QAA [2]_
code are based on the original code written in Matlab.

.. [1] Cardoso, J. F.; Souloumiac, A. "Blind Beamforming for Non-Gaussian
       Signals." IEE Proc F Radar Signal Process 1993, 140 (6), 362.
.. [2] Ramanathan, A.; Savol, A. J.; Langmead, C. J.; Agarwal, P. K.;
       Chennubhotla, C. S. "Discovering Conformational Sub-States Relevant to Protein
       Function." Plos One 2011, 6 (1), e15827.

Requirements
------------

* Python 3.8+
* click 7.0+
* numpy 1.20+
* scipy 1.6+
* matplotlib 3.3+
* scikit-learn 0.24+
* mdtraj 1.9+
* nptyping 1.4+
* holoviews 1.14+

Installation
------------

You can install *Quasi-Anharmonic Analysis* via pip_ from PyPI_:

.. code:: console

   $ pip install qaa

If you want to visualize the tutorial notebooks, you can install the extra
dependencies via pip_ from PyPI_:

.. code:: console

   $ pip install qaa[jupyter]


Usage
-----

Please see the `Command-line Reference <Usage_>`_ for details.


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `BSD 3 Clause license`_,
*Quasi-Anharmonic Analysis* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _BSD 3 Clause license: https://opensource.org/licenses/BSD-3-Clause
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/tclick/qaa/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://qaa.readthedocs.io/en/latest/usage.html
