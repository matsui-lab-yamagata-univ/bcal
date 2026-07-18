==========================================================================================================
bcal: Program for the calculation of band structure and effective mass for organic semiconductor crystals
==========================================================================================================

.. image:: https://img.shields.io/badge/python-3.11%20or%20newer-blue
   :target: https://www.python.org
   :alt: Python

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Overview
========

``bcal`` is a tool for calculating the band structure and effective masses of
organic semiconductors. Starting from a crystal structure (CIF), it generates
quantum-chemistry inputs, runs DFT calculations, extracts the molecular-orbital
matrices, and builds a tight-binding Hamiltonian. From the resulting band
dispersion it reports the HOMO and LUMO band edges together with their
principal effective masses and axis vectors, and draws the band diagram along a
configurable high-symmetry k-path.

.. toctree::
   :maxdepth: 2

   guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   bcal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
