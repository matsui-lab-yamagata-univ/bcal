==========
User Guide
==========

Installation
============

Requirements
------------

* Python 3.11 or newer
* NumPy
* Pandas
* SciPy
* Matplotlib
* yu-mcal>=0.7.1
* yu-tcal>=5.0.2

Quantum Chemistry Calculation Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At least one of the following is required:

* Gaussian 09 or 16
* PySCF (macOS / Linux / WSL2(Windows Subsystem for Linux))
* GPU4PySCF (macOS / Linux / WSL2(Windows Subsystem for Linux))
* ORCA 6.1.0 or newer

Important notice
----------------

* When using Gaussian, the path of the Gaussian executable must be set.
* PySCF is supported on macOS / Linux. Windows users must use WSL2.

Installing bcal
---------------

Install from PyPI, choosing the extra that matches the backend you intend to
use.

Using Gaussian 09 or 16 (without PySCF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install yu-bcal

Using PySCF (CPU only, macOS / Linux / WSL2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install "yu-bcal[pyscf]"

Using GPU acceleration with PySCF (macOS / Linux / WSL2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Check your installed CUDA Toolkit version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   nvcc --version

2. Install the GPU extra that matches your CUDA Toolkit version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your CUDA Toolkit version is 13.x:

.. code-block:: bash

   pip install "yu-bcal[gpu4pyscf-cuda13]"

If your CUDA Toolkit version is 12.x:

.. code-block:: bash

   pip install "yu-bcal[gpu4pyscf-cuda12]"

If your CUDA Toolkit version is 11.x:

.. code-block:: bash

   pip install "yu-bcal[gpu4pyscf-cuda11]"

Using ORCA 6.1.0 or newer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install "yu-bcal[orca]"

Verify Installation
-------------------

After installation, you can verify by running:

.. code-block:: bash

   bcal --help

Usage
=====

Basic Usage
-----------

.. code-block:: bash

   bcal <cif_filename> [options]

Required Arguments
~~~~~~~~~~~~~~~~~~~

- ``file``: Path to the CIF file.

Unlike mobility-tensor tools, ``bcal`` does **not** take a ``p``/``n``
semiconductor type. A single run reports **both** the HOMO band edge (relevant
for p-type transport) and the LUMO band edge (relevant for n-type transport).

Basic Examples
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Compute band structure and effective masses for a crystal
   bcal xxx.cif

   # Same, using PySCF as the backend
   bcal xxx.cif --engine pyscf

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 8 28 64

   * - Short
     - Long
     - Explanation
   * - ``-h``
     - ``--help``
     - Show options description.
   * - ``-M``
     - ``--method METHOD/BASIS``
     - Calculation method and basis set in "METHOD/BASIS" format. (default: ``PBEPBE/6-31G(d,p)``)
   * - ``-c``
     - ``--cpu N``
     - Set the number of CPUs. (default: ``4``)
   * - ``-m``
     - ``--mem N``
     - Set the memory size in GB. (default: ``10``)
   * - ``-o``
     - ``--output DIR``
     - Output directory for the results. (default: the directory of the input CIF file)
   * - ``-r``
     - ``--read``
     - Read existing log files without executing calculations.
   * -
     - ``--engine ENGINE``
     - Quantum-chemistry backend: ``g16``, ``g09``, ``pyscf``, ``gpu4pyscf``, ``orca``. (default: ``g16``)
   * -
     - ``--resume``
     - Resume an interrupted calculation from the last incomplete step.
   * -
     - ``--num-mo N``
     - Number of MOs per molecule on each side of the frontier; total MOs = ``2 * N``. (default: ``15``)
   * -
     - ``--band-path PATH``
     - High-symmetry k-path for the band diagram, given as single-character labels (``G`` = Gamma). (default: ``XGYGZ``)
   * -
     - ``--bse``
     - Use Basis Set Exchange to obtain basis sets. (``pyscf``/``gpu4pyscf`` only)

Calculation Settings
~~~~~~~~~~~~~~~~~~~~~~

``-M, --method <method>``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify the DFT method and basis set used in the quantum-chemistry calculations.

- **Default**: ``PBEPBE/6-31G(d,p)``
- **Example**: ``bcal xxx.cif -M "B3LYP/6-31G(d,p)"``

.. note::

   ``PBEPBE`` is the Gaussian spelling of the PBE functional. The ``pyscf``,
   ``gpu4pyscf``, and ``orca`` engines do not accept this name, so ``bcal``
   automatically rewrites a leading ``PBEPBE`` to ``PBE`` (e.g.
   ``PBEPBE/6-31G(d,p)`` → ``PBE/6-31G(d,p)``) and emits a warning to stderr.

``-c, --cpu <number>``
^^^^^^^^^^^^^^^^^^^^^^^

Specify the number of CPUs to use.

- **Default**: ``4``
- **Example**: ``bcal xxx.cif -c 8``

``-m, --mem <memory>``
^^^^^^^^^^^^^^^^^^^^^^^

Specify the amount of memory in GB.

- **Default**: ``10``
- **Example**: ``bcal xxx.cif -m 16``

``--engine <engine>``
^^^^^^^^^^^^^^^^^^^^^^

Specify the quantum-chemistry backend.

- **Choices**: ``g16``, ``g09``, ``pyscf``, ``gpu4pyscf``, ``orca``
- **Default**: ``g16`` (Gaussian 16)
- **Examples**:

  - ``bcal xxx.cif --engine g09`` (Gaussian 09)
  - ``bcal xxx.cif --engine pyscf`` (PySCF, CPU; requires the ``pyscf`` extra)
  - ``bcal xxx.cif --engine gpu4pyscf`` (GPU-accelerated PySCF; requires a ``gpu4pyscf-cudaXX`` extra)
  - ``bcal xxx.cif --engine orca`` (ORCA; requires the ``orca`` extra)

ORCA parallel execution
"""""""""""""""""""""""

ORCA is driven through the ORCA Python Interface (OPI). To use multiple CPU
cores (``--cpu N``), OpenMPI must be installed and visible to ORCA.
First, confirm that ``mpirun`` is available:

.. code-block:: bash

   which mpirun

If OpenMPI is already in ``$PATH`` and ``$LD_LIBRARY_PATH`` (common on Linux/WSL
after ``apt install``), no further configuration is usually needed. Otherwise,
point ORCA at the OpenMPI base directory (the directory that contains ``bin/``
and ``lib/``) via the ``OPI_MPI`` environment variable:

.. code-block:: bash

   # Built from source or via a module system
   which mpirun
   # e.g. /opt/openmpi/bin/mpirun  ->  base: /opt/openmpi
   export OPI_MPI=$(dirname $(dirname $(which mpirun)))

   # Installed system-wide via apt (Ubuntu/Debian)
   export OPI_MPI=/usr/lib/x86_64-linux-gnu/openmpi

.. note::

   ORCA requires a specific version of OpenMPI. The version available via
   ``apt`` may not match. If parallel execution fails, build OpenMPI from
   source using the version specified in the `ORCA documentation
   <https://www.faccts.de/docs/orca/6.0/manual/>`_.

``--bse``
^^^^^^^^^

Resolve the basis set named in ``-M, --method`` through `Basis Set Exchange
<https://www.basissetexchange.org/>`_ instead of the engine's built-in
definition. This is a flag (it takes no value) and **only affects the**
``pyscf`` **and** ``gpu4pyscf`` **engines** (it uses the ``basis-set-exchange``
package bundled with the ``pyscf`` extra); it is silently ignored by ``g16``,
``g09``, and ``orca``.

- **Default**: off
- **Example**: ``bcal xxx.cif --engine pyscf --bse``

Band-Structure Settings
~~~~~~~~~~~~~~~~~~~~~~~~~

``--num-mo <number>``
^^^^^^^^^^^^^^^^^^^^^^

Number of molecular orbitals (MOs) per molecule kept on each side of the
frontier (HOMO side and LUMO side); the total number of MOs used per molecule
is ``2 * num_mo``. Increasing it widens the orbital window included in the
tight-binding model.

- **Default**: ``15``
- **Example**: ``bcal xxx.cif --num-mo 20``

``--band-path <path>``
^^^^^^^^^^^^^^^^^^^^^^^

High-symmetry k-path for the band diagram, given as a sequence of
single-character labels (``G`` = Gamma).

- **Default**: ``XGYGZ`` (traces X → Γ → Y → Γ → Z)
- **Example**: ``bcal xxx.cif --band-path GXSY``

Available high-symmetry points:

.. list-table::
   :header-rows: 1

   * - Label
     - Fractional reciprocal coordinates
   * - ``G``
     - (0.0, 0.0, 0.0) — Gamma
   * - ``X``
     - (0.5, 0.0, 0.0)
   * - ``Y``
     - (0.0, 0.5, 0.0)
   * - ``Z``
     - (0.0, 0.0, 0.5)
   * - ``S``
     - (0.5, 0.5, 0.0)
   * - ``T``
     - (0.0, 0.5, 0.5)
   * - ``U``
     - (0.5, 0.0, 0.5)
   * - ``R``
     - (0.5, 0.5, 0.5)

The chosen path only affects the band diagram (``band.png``); the band edges and
effective masses are searched over the full Brillouin zone. If a HOMO/LUMO edge
does not lie on the requested path, ``bcal`` emits a warning to stderr so you can
add the corresponding k-point to ``--band-path``.

Output Settings
~~~~~~~~~~~~~~~~

``-o, --output <directory>``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output directory for the results. A per-crystal subdirectory named after the
CIF is created inside it.

- **Default**: the directory containing the input CIF file.
- **Example**: ``bcal xxx.cif -o ./results``

Calculation Control
~~~~~~~~~~~~~~~~~~~~~

``-r, --read``
^^^^^^^^^^^^^^^

Read results from existing log files without executing new DFT calculations.
Skips input generation and reuses the logs already present under the crystal's
``logs/`` directory.

- **Example**: ``bcal xxx.cif -r``

``--resume``
^^^^^^^^^^^^^

Resume an interrupted calculation, reusing completed steps and continuing from
the last incomplete one (e.g. DFT calculations that already terminated normally
are not repeated).

- **Example**: ``bcal xxx.cif --resume``

Practical Usage Examples
------------------------

Basic Calculations
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Default run (Gaussian 16, PBEPBE/6-31G(d,p))
   bcal xxx.cif

   # Use 8 CPUs and 16 GB memory
   bcal xxx.cif -c 8 -m 16

Choosing a Backend
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # PySCF (CPU)
   bcal xxx.cif --engine pyscf

   # GPU-accelerated PySCF
   bcal xxx.cif --engine gpu4pyscf

   # ORCA with 8 CPUs
   bcal xxx.cif --engine orca -c 8

   # PySCF with basis sets from Basis Set Exchange
   bcal xxx.cif --engine pyscf --bse

Tuning the Band Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Different method / basis set
   bcal xxx.cif -M "B3LYP/6-311G(d,p)"

   # Wider orbital window
   bcal xxx.cif --num-mo 25

   # Custom k-path for the band diagram
   bcal xxx.cif --band-path GXSYG

Reusing Results
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Read from existing log files (no new DFT)
   bcal xxx.cif -r

   # Resume an interrupted calculation
   bcal xxx.cif --resume

Output
======

Standard Output
---------------

Results are written to **stdout**, while all warnings and diagnostics are routed
to **stderr** (prefixed with ``WARNING:``, coloured when stderr is an interactive
terminal). The two streams can therefore be redirected independently.

At the end of a run, ``bcal`` prints to stdout, for each frontier band:

- The **band edge**: its high-symmetry label (when applicable) and fractional
  reciprocal-lattice coordinates.
- The three **principal effective masses** ``m1``, ``m2``, ``m3`` (in units of
  the electron mass ``m_e``), sorted by ``|m|`` in ascending order, each paired
  with its unit principal-axis vector in Cartesian reciprocal space.

.. code-block:: text

   LUMO band edge: G  k=(+0.000, +0.000, +0.000)
     m1 = +0.834 m_e  v=(+0.998, +0.043, +0.000)
     m2 = +1.207 m_e  v=(-0.043, +0.998, +0.000)
     m3 = +3.115 m_e  v=(+0.000, +0.000, +1.000)

   HOMO band edge: X  k=(+0.500, +0.000, +0.000)
     m1 = -0.756 m_e  v=(+0.991, +0.000, +0.135)
     m2 = -1.042 m_e  v=(+0.000, +1.000, +0.000)
     m3 = -2.880 m_e  v=(-0.135, +0.000, +0.991)

A warning is printed to stderr if a band edge falls outside the requested
``--band-path``; the path to the saved results and the elapsed time follow on
stdout.

Generated Files
---------------

``bcal`` writes a self-contained directory per crystal, with the pipeline stages
kept in separate subdirectories:

.. code-block:: text

   <output>/<NAME>/
   ├── structure.json     # Crystal topology metadata (lattice, method, engine,
   │                      #   sites, dimer_types, pairs) — human-readable
   ├── inputs/            # Generated QM input files
   │   ├── <NAME>_monomer_000.gjf   # .gjf for g16/g09, .xyz for pyscf/gpu4pyscf/orca
   │   ├── <NAME>_dimer_000.gjf
   │   └── ...
   ├── logs/              # DFT output logs (.log / .out, .chk)
   ├── matrices/          # Extracted numerics (NumPy .npz)
   │   ├── monomers.npz   # Monomer MO coefficients and energy levels
   │   ├── dimers.npz     # Dimer overlap / Fock matrices
   │   └── transfer.npz   # Transfer integrals and on-site levels
   └── results/           # Final tight-binding results
       ├── band.png       # Band structure diagram along --band-path
       ├── band.npz       # Band energies, k-distances, tick positions and labels
       └── effective_mass.csv  # HOMO/LUMO band-edge effective masses and axis vectors

``structure.json``
~~~~~~~~~~~~~~~~~~~

Human-readable record of the generated structure: ``name``, ``method``,
``engine``, generation parameters, the direct ``lattice`` vectors, the unique
molecular ``sites``, the symmetry-unique ``dimer_types``, and the full
``pairs`` table (molecule indices, lattice offsets, and the dimer type each pair
maps to). Per-calculation numerics are stored separately under ``matrices/``.

Input file naming
~~~~~~~~~~~~~~~~~~

Monomer and dimer inputs are named ``<NAME>_monomer_{index:03d}`` and
``<NAME>_dimer_{type:03d}`` (with the extension chosen by the engine). The
filenames carry only the site/dimer-type identifier; the mapping information
needed to reassemble the tight-binding model (central/neighbor molecules,
lattice offsets, orbital ordering) lives in ``structure.json``.

Notes & Troubleshooting
=======================

Notes
-----

#. **Calculation Time**: Calculation time depends strongly on the number of
   molecules per cell, the method/basis set, and the chosen backend.
#. **Memory Usage**: Ensure sufficient memory for large systems (``-m``).
#. **Gaussian Installation**: Gaussian 09 or Gaussian 16 is required for the
   ``g09`` / ``g16`` engines.
#. **Dependencies**: Make sure the optional dependencies for your chosen backend
   are installed (see `Installation`_).

Troubleshooting
---------------

If a calculation stops midway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Resume with the --resume option
   bcal xxx.cif --resume

Memory shortage error
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Increase the amount of memory
   bcal xxx.cif -m 32

A band edge is not on the band diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``bcal`` warns that a HOMO/LUMO edge is not on ``--band-path``, the band edge
and effective masses are still correct (they are found over the full Brillouin
zone), but ``band.png`` will not show the true extremum. Add the reported
k-point's high-symmetry label to ``--band-path`` and re-run with ``-r`` to
redraw without recomputing the DFT.

If a CIF file cannot be read
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CIF files come in various formats, and some may not be readable by bcal. Please
try the following:

#. **Convert the CIF format using another software**: Use software such as
   `Mercury <https://www.ccdc.cam.ac.uk/solutions/software/mercury/>`_ to open
   the CIF file and re-export it, which may resolve the issue.
#. **Contact us**: If you send the unreadable CIF file to us by email, we will
   work on adding support for it. Please contact us at the email address listed
   in `Authors & References`_.

Authors & References
====================

Authors
-------

`Matsui Laboratory, Research Center for Organic Electronics (ROEL), Yamagata University <https://matsui-lab.yz.yamagata-u.ac.jp/index-e.html>`_

Tomoharu Okada, Koki Ozawa, Yu Homma, Hiroyuki Matsui

Email: h-matsui[at]yz.yamagata-u.ac.jp

Please replace [at] with @

References
----------

[1] Qiming Sun et al., Recent developments in the PySCF program package,
*J. Chem. Phys.* **2020**, *153*, 024109.

[2] Benjamin P. Pritchard et al., New Basis Set Exchange: An Open, Up-to-Date
Resource for the Molecular Sciences Community, *J. Chem. Inf. Model.* **2019**,
*59*, 4814-4820.

[3] Frank Neese, The ORCA program system, *Wiley Interdiscip. Rev. Comput. Mol.
Sci.*, **2012**, *2*, 73-78.
