"""Band structure and effective mass from a generated structure directory.

This module consumes the per-crystal layout produced by
:class:`bcal.utils.input_maker.InputMaker` (``structure.json`` + ``inputs/``)
and runs the calculation pipeline:

``inputs/`` --(DFT)--> ``logs/`` --(extract)--> ``matrices/`` --(TB)--> ``results/``

The crystal-symmetry transfer-integral construction (one DFT per unique dimer
type, reused for symmetry-equivalent pairs) is the core of bcal and is ported
from the reference implementation. The external library ``yu-tcal`` is reused
for SCF orchestration (``TcalPySCF``/``TcalORCA`` driven with
``skip_monomer_num``), Gaussian log/matrix parsing (``Tcal.read_matrix`` etc.),
and the transfer-integral formula/unit (``Tcal.EV``).
"""
from __future__ import annotations

import io
import itertools
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from scipy.optimize import fmin
from tcal import Tcal

from bcal.utils.log import get_logger

logger = get_logger(__name__)

HARTREE_TO_EV: float = 27.211386245988
GAUSSIAN_ENGINES = {"g16", "g09"}
PYSCF_ENGINES = {"pyscf", "gpu4pyscf"}

# High-symmetry point labels -> fractional reciprocal-lattice coordinates.
# Each label is a single character; "G" denotes Gamma. A band-path string such
# as "XGYGZ" is read one character at a time by :meth:`Bcal.draw_band_diagram`.
HIGH_SYMMETRY_POINTS: dict[str, tuple[float, float, float]] = {
    "G": (0.0, 0.0, 0.0),   # Gamma
    "X": (0.5, 0.0, 0.0),
    "Y": (0.0, 0.5, 0.0),
    "Z": (0.0, 0.0, 0.5),
    "S": (0.5, 0.5, 0.0),
    "T": (0.0, 0.5, 0.5),
    "U": (0.5, 0.0, 0.5),
    "R": (0.5, 0.5, 0.5),
}

# Reverse lookup of HIGH_SYMMETRY_POINTS: fractional coordinates -> label.
_LABEL_BY_FRAC: dict[tuple[float, float, float], str] = {
    coord: label for label, coord in HIGH_SYMMETRY_POINTS.items()
}


class Bcal:
    """Run the band-structure / effective-mass pipeline for one structure.

    Parameters
    ----------
    struct_dir : str or pathlib.Path
        Directory containing ``structure.json`` and ``inputs/`` (as written by
        :class:`bcal.utils.input_maker.InputMaker`). ``logs/``, ``matrices/``
        and ``results/`` are created beneath it.
    num_mo : int, optional
        Number of MOs per molecule on each side of the frontier; the total
        window is ``n = 2 * num_mo`` orbitals (``num_mo`` HOMO-side and
        ``num_mo`` LUMO-side). By default 15.
    engine : str, optional
        Calculation engine and the single source of engine selection:
        ``"g16"``/``"g09"`` (Gaussian), ``"pyscf"``/``"gpu4pyscf"`` (PySCF),
        ``"orca"`` (ORCA). By default ``"g16"``.
    cpu : int, optional
        Number of CPU cores / threads for the engine, by default 4.
    mem : int, optional
        Memory in GB for the engine, by default 16.
    method : str, optional
        DFT method/basis string, by default ``"B3LYP/6-31G(d,p)"``.
    bse : bool, optional
        Use Basis Set Exchange to resolve the basis set (PySCF/gpu4pyscf
        only; ignored for other engines), by default ``False``.
    """

    EV: float = Tcal.EV

    def __init__(
        self,
        struct_dir: str | Path,
        num_mo: int = 15,
        engine: str = "g16",
        cpu: int = 4,
        mem: int = 16,
        method: str = "B3LYP/6-31G(d,p)",
        bse: bool = False,
    ) -> None:
        self.struct_dir = Path(struct_dir)
        self.inputs_dir = self.struct_dir / "inputs"
        self.logs_dir = self.struct_dir / "logs"
        self.matrices_dir = self.struct_dir / "matrices"
        self.results_dir = self.struct_dir / "results"

        self.engine = engine
        self.cpu = cpu
        self.mem = mem
        self.method = method
        self.bse = bse
        self.num_mo = num_mo
        self.n = 2 * num_mo

        with open(self.struct_dir / "structure.json") as f:
            structure = json.load(f)
        self.name: str = structure["name"]
        self.lattice = np.asarray(structure["lattice"], dtype=np.float64)
        self.sites = structure["sites"]
        self.dimer_types = structure["dimer_types"]
        self.m = len(self.sites)
        ref_symbols = self.sites[0]["symbols"]
        for site in self.sites[1:]:
            if site["symbols"] != ref_symbols:
                raise ValueError(
                    "All unique molecules must be the same chemical species "
                    "(identical atom count and ordering); bcal does not support "
                    "co-crystals or mixed species. Offending site "
                    f"idx={site['idx']}."
                )
        self.n_atoms = len(self.sites[0]["symbols"])
        self.num_dimer_types = len(self.dimer_types)

        self.pairs, self.atoms_order = self._build_pairs(structure["pairs"])
        self.n_pairs = self.pairs.shape[0]

        # DFT-derived quantities (filled by run_dft / _load_matrices).
        self.MOs: Optional[np.ndarray] = None
        self.energies: Optional[np.ndarray] = None
        self.n_elect: Optional[int] = None
        self.n_basis: Optional[int] = None
        self.n_basis_d: Optional[int] = None
        self.ao_atom: Optional[np.ndarray] = None
        self.ao_xyz: Optional[np.ndarray] = None
        self.overlap_ori: Optional[np.ndarray] = None
        self.fock_ori: Optional[np.ndarray] = None
        self.levels: Optional[np.ndarray] = None
        self.trans: Optional[np.ndarray] = None
        self.band_energies: Optional[np.ndarray] = None

        # Band-edge k-points (filled by cal_effective_mass): fractional
        # reciprocal-lattice coordinates and the high-symmetry label (or None).
        self.homo_edge_frac: Optional[np.ndarray] = None
        self.lumo_edge_frac: Optional[np.ndarray] = None
        self.homo_edge_label: Optional[str] = None
        self.lumo_edge_label: Optional[str] = None

        # Principal effective masses and their axes (filled by
        # cal_effective_mass): masses sorted by |m| ascending and the matching
        # unit principal-axis vectors as rows (Cartesian reciprocal-space).
        self.homo_em: Optional[np.ndarray] = None
        self.lumo_em: Optional[np.ndarray] = None
        self.homo_em_vectors: Optional[np.ndarray] = None
        self.lumo_em_vectors: Optional[np.ndarray] = None

        self._autos_cache: dict = {}
        self._hermitian_warned = False

        self._ext_log = "out" if platform.system().lower() == "windows" else "log"

    # ------------------------------------------------------------------ #
    # structure.json -> arrays                                           #
    # ------------------------------------------------------------------ #
    def _build_pairs(self, pairs: list[dict]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Reconstruct the integer ``pairs`` array and per-pair atom mapping.

        Parameters
        ----------
        pairs : list of dict
            ``structure.json`` ``pairs[]`` entries.

        Returns
        -------
        pairs_arr : numpy.ndarray, shape (n_pairs, 10)
            Columns ``[cent, nei, i, j, k, ti_flg, transpose, sx, sy, sz]``.
        atoms_order : numpy.ndarray or None
            Per-pair atom permutation, shape (n_pairs, 2*n_atoms). ``None`` if
            every pair uses the identity ordering (no reordering needed).
        """
        n_pairs = len(pairs)
        pairs_arr = np.zeros((n_pairs, 10), dtype=np.int64)
        atoms_order = np.empty((n_pairs, 2 * self.n_atoms), dtype=np.int64)
        identity = np.tile(np.arange(self.n_atoms, dtype=np.int64), 2)
        any_sort = False
        for idx, p in enumerate(pairs):
            pairs_arr[idx] = [
                p["cent"], p["nei"], p["ijk"][0], p["ijk"][1], p["ijk"][2],
                p["ti_flg"], p["transpose"], p["signs"][0], p["signs"][1], p["signs"][2],
            ]
            if p.get("need_sort", 0) == 1 and "atoms_order" in p:
                atoms_order[idx] = np.asarray(p["atoms_order"], dtype=np.int64)
                any_sort = True
            else:
                atoms_order[idx] = identity
        return pairs_arr, (atoms_order if any_sort else None)

    # ------------------------------------------------------------------ #
    # DFT driver                                                         #
    # ------------------------------------------------------------------ #
    def run_dft(self, read: bool = False, resume: bool = False) -> None:
        """Execute (or read) DFT, extract matrices, and build transfer integrals.

        Parameters
        ----------
        read : bool, optional
            If True, do not run SCF. Reuse ``matrices/`` if present, otherwise
            parse existing engine outputs (Gaussian ``logs/``, PySCF chkfiles).
            By default False.
        resume : bool, optional
            If True, skip inputs whose engine output already completed, by
            default False.
        """
        self.matrices_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if read and self._load_matrices():
            print(f"Loaded matrices from {self.matrices_dir}")
        else:
            self._dft_header(run_scf=not read)
            if self.engine in GAUSSIAN_ENGINES:
                self._run_dft_gaussian(run_scf=not read, resume=resume)
            elif self.engine in PYSCF_ENGINES:
                self._run_dft_pyscf(run_scf=not read, resume=resume)
            elif self.engine == "orca":
                self._run_dft_orca(run_scf=not read, resume=resume)
            else:
                raise ValueError(f"Unknown engine: {self.engine}")
            self._save_matrices()
            self._dft_footer()

        n_orb = self.energies.shape[1]
        if self.num_mo > self.n_elect or self.n_elect + self.num_mo > n_orb:
            raise ValueError(
                f"num_mo={self.num_mo} exceeds the available orbital window: "
                f"the molecule has {self.n_elect} occupied and "
                f"{n_orb - self.n_elect} virtual orbital(s), and num_mo must not "
                f"exceed either. Reduce --num-mo (e.g. "
                f"<= {min(self.n_elect, n_orb - self.n_elect)})."
            )

        self.levels = self.energies[:, self.n_elect - self.n // 2 : self.n_elect + self.n // 2]
        self.cal_transfer_integrals_all()
        np.savez_compressed(self.matrices_dir / "transfer.npz", trans=self.trans, levels=self.levels)

    def _load_matrices(self) -> bool:
        """Load previously extracted ``monomers.npz``/``dimers.npz`` if present.

        Returns
        -------
        bool
            ``True`` if both files existed and were loaded.
        """
        mono = self.matrices_dir / "monomers.npz"
        dim = self.matrices_dir / "dimers.npz"
        if not (mono.exists() and dim.exists()):
            return False
        m = np.load(mono)
        self.MOs = m["mo"]
        self.energies = m["energies"]
        self.n_elect = int(m["n_elect"])
        self.n_basis = int(m["n_basis"])
        self.ao_atom = m["ao_atom"]
        self.ao_xyz = m["ao_xyz"]
        d = np.load(dim)
        self.overlap_ori = d["overlap"]
        self.fock_ori = d["fock"]
        self.n_basis_d = self.overlap_ori.shape[-1]
        return True

    def _save_matrices(self) -> None:
        """Write extracted monomer/dimer matrices to ``matrices/``."""
        np.savez_compressed(
            self.matrices_dir / "monomers.npz",
            mo=self.MOs,
            energies=self.energies,
            n_elect=self.n_elect,
            n_basis=self.n_basis,
            ao_atom=self.ao_atom,
            ao_xyz=self.ao_xyz,
        )
        np.savez_compressed(
            self.matrices_dir / "dimers.npz",
            overlap=self.overlap_ori,
            fock=self.fock_ori,
        )

    # ------------------------------------------------------------------ #
    # DFT progress reporting (uniform across engines)                    #
    # ------------------------------------------------------------------ #
    def _dft_header(self, run_scf: bool) -> None:
        """Print the one-line header that precedes the per-job DFT progress.

        Parameters
        ----------
        run_scf : bool
            ``True`` when SCF is executed, ``False`` when existing engine
            outputs are only re-read.
        """
        verb = "Running DFT" if run_scf else "Reading DFT logs"
        print()  # blank line separating the input-generation output from the DFT log
        print(f"{verb} (engine: {self.engine}, method: {self.method})")
        print()  # blank line separating the header from the first job block

    def _dft_footer(self) -> None:
        """Print the closing summary line after all DFT jobs completed."""
        total = self.m + self.num_dimer_types
        print(f"DFT done: {total} jobs completed, matrices -> {self.matrices_dir}")

    def _report_job(self, kind: str, pos: int, total: int, stem: str, status: str) -> None:
        """Print a single DFT job's progress line to stdout.

        Parameters
        ----------
        kind : str
            Job kind, ``"monomer"`` or ``"dimer"``.
        pos : int
            1-based index of the job within its kind.
        total : int
            Number of jobs of this kind.
        stem : str
            Input file stem of the job (e.g. ``"<name>_monomer_000"``).
        status : str
            Outcome label (``"running"``, ``"done"``, ``"skip (resume)"``, ...).
        """
        print(f"{kind:<7} {pos}/{total}  {stem} ... {status}")

    def _begin_job(self, kind: str, pos: int, total: int, stem: str) -> None:
        """Print the start-of-calculation line for a DFT job (before reading)."""
        self._report_job(kind, pos, total, stem, "running")

    @staticmethod
    def _reading(path: Path) -> None:
        """Print the line announcing which engine output is being parsed.

        Emitted by bcal only for the read paths where the underlying
        ``yu-tcal`` reader is silent (Gaussian logs parsed directly, and the
        PySCF dimer loaded via ``_load_from_chk_dimer``); the PySCF monomer and
        all ORCA reads print their own ``reading ...`` line, so bcal does not
        duplicate it there.
        """
        print(f"reading {path}")

    def _finish_job(
        self, kind: str, pos: int, total: int, stem: str, ok: bool, fail_status: str = "FAILED"
    ) -> None:
        """Report a job's outcome and abort the run if it did not succeed.

        Parameters
        ----------
        kind, pos, total, stem
            See :meth:`_report_job`.
        ok : bool
            ``True`` if the job terminated normally with a converged SCF.
        fail_status : str, optional
            Status label printed when ``ok`` is ``False``, by default
            ``"FAILED"``.

        Raises
        ------
        RuntimeError
            If ``ok`` is ``False``; the run does not continue past a job that
            did not complete successfully.
        """
        self._report_job(kind, pos, total, stem, "done" if ok else fail_status)
        if not ok:
            logger.error(f"DFT job {stem} did not complete successfully ({fail_status}); aborting.")
            raise RuntimeError(f"DFT job {stem} did not complete successfully ({fail_status}).")
        print()  # blank line separating consecutive job blocks

    def _skip_job(self, kind: str, pos: int, total: int, stem: str) -> None:
        """Report a job reused from a previous run (``--resume``) and separate it."""
        self._report_job(kind, pos, total, stem, "skip (resume)")
        print()  # blank line separating consecutive job blocks

    # ------------------------------------------------------------------ #
    # Gaussian engine                                                    #
    # ------------------------------------------------------------------ #
    def _run_dft_gaussian(self, run_scf: bool, resume: bool) -> None:
        """Run/parse Gaussian for all monomers and dimer types.

        Parameters
        ----------
        run_scf : bool
            If True, run Gaussian; otherwise only parse existing logs.
        resume : bool
            If True, skip inputs whose log already terminated normally.
        """
        cmd = "g09" if self.engine == "g09" else "g16"

        n_elect_arr = np.empty(self.m, dtype=np.int64)
        for pos, site in enumerate(self.sites, start=1):
            a = int(site["idx"])
            stem = f"{self.name}_monomer_{a:03d}"
            log = self.logs_dir / f"{stem}.{self._ext_log}"
            skipped = self._run_gaussian_input("monomer", pos, self.m, stem, cmd, log, run_scf, resume)
            self._reading(log)
            mo, ene, n_elect, orbital_str = self._parse_gaussian_monomer(log)
            if skipped:
                self._skip_job("monomer", pos, self.m, stem)
            else:
                self._finish_job("monomer", pos, self.m, stem, self._gaussian_terminated(log))
            if self.MOs is None:
                nb = mo.shape[0]
                self.MOs = np.empty((self.m, nb, mo.shape[1]))
                self.energies = np.empty((self.m, ene.shape[0]))
                self.ao_atom, self.ao_xyz = self._orbital_str_to_ao_meta(orbital_str)
                self.n_basis = nb
            self.MOs[a] = mo
            self.energies[a] = ene * HARTREE_TO_EV
            n_elect_arr[a] = n_elect

        for pos, dt in enumerate(self.dimer_types, start=1):
            i = int(dt["ti_flg"])
            stem = f"{self.name}_dimer_{i:03d}"
            log = self.logs_dir / f"{stem}.{self._ext_log}"
            skipped = self._run_gaussian_input("dimer", pos, self.num_dimer_types, stem, cmd, log, run_scf, resume)
            self._reading(log)
            ov, fk = self._parse_gaussian_dimer(log)
            if skipped:
                self._skip_job("dimer", pos, self.num_dimer_types, stem)
            else:
                self._finish_job("dimer", pos, self.num_dimer_types, stem, self._gaussian_terminated(log))
            if self.overlap_ori is None:
                nbd = ov.shape[0]
                self.overlap_ori = np.empty((self.num_dimer_types, nbd, nbd))
                self.fock_ori = np.empty((self.num_dimer_types, nbd, nbd))
                self.n_basis_d = nbd
            self.overlap_ori[i] = ov
            self.fock_ori[i] = fk

        if not (n_elect_arr == n_elect_arr[0]).all():
            raise ValueError(f"Inconsistent electron counts across monomers: {n_elect_arr}")
        self.n_elect = int(n_elect_arr[0])

    def _run_gaussian_input(
        self, kind: str, pos: int, total: int, stem: str, cmd: str, log: Path, run_scf: bool, resume: bool
    ) -> bool:
        """Run a single Gaussian input in ``logs/`` (or verify its log exists).

        Prints the ``running`` start line (via :meth:`_begin_job`) immediately
        before launching the engine, so it appears ahead of the subsequent
        ``reading`` line.

        Returns
        -------
        bool
            ``True`` if the job was skipped because a completed log already
            exists (``resume``); ``False`` if it was run or only verified.
        """
        if not run_scf:
            if not log.exists():
                raise FileNotFoundError(f"{log} not found (read mode). Run without read first.")
            return False
        if resume and self._gaussian_terminated(log):
            return True
        self._begin_job(kind, pos, total, stem)
        shutil.copy(self.inputs_dir / f"{stem}.gjf", self.logs_dir / f"{stem}.gjf")
        self._execute([cmd, f"{stem}.gjf"], cwd=self.logs_dir)
        return False

    @staticmethod
    def _gaussian_terminated(log: Path) -> bool:
        """Return True if the Gaussian log ends with a normal termination."""
        if not log.exists():
            return False
        with open(log, "rb") as f:
            try:
                f.seek(-2048, 2)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="ignore")
        return "Normal termination" in tail

    def _parse_gaussian_monomer(
        self, log: Path
    ) -> Tuple[np.ndarray, np.ndarray, int, list[str]]:
        """Extract MO coefficients, orbital energies, n_elect, and AO labels.

        Returns
        -------
        mo : numpy.ndarray, shape (n_basis, n_basis)
            Square MO coefficient matrix (rows = AO, cols = MO).
        energies : numpy.ndarray, shape (n_orbitals,)
            Orbital energies in Hartree.
        n_elect : int
            Number of doubly occupied orbitals (= number of beta electrons).
        orbital_str : list of str
            Cartesian AO labels (e.g. ``"1S"``, ``"2PX"``, ``"4XX"``).
        """
        lines = self._read_log_lines(log)
        second = self._find(lines, "Normal termination") + 1
        n_basis = n_elect = mo = orbital_str = None
        for i in range(second, len(lines)):
            line = lines[i]
            if n_basis is None:
                n_basis = Tcal.extract_num("[0-9]+ cartesian basis functions", line)
            if n_elect is None:
                n_elect = Tcal.extract_num("[0-9]+ beta electrons", line)
            if "Alpha MO coefficients at cycle" in line:
                mo = Tcal.read_matrix(io.StringIO("".join(lines[i + 1 :])), n_basis, n_basis)
            if "Molecular Orbital Coefficients:" in line:
                orbital_str = self._read_orbital_str(lines, i, n_basis)
                break
        if mo is None or orbital_str is None:
            raise ValueError(f"Failed to parse MO coefficients/labels from {log}")
        energies = self._extract_energy(lines)
        return mo, energies, int(n_elect), orbital_str

    def _parse_gaussian_dimer(self, log: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Extract dimer overlap and Fock matrices from a Gaussian log.

        Returns
        -------
        overlap : numpy.ndarray, shape (n_basis_d, n_basis_d)
        fock : numpy.ndarray, shape (n_basis_d, n_basis_d)
        """
        lines = self._read_log_lines(log)
        second = self._find(lines, "Normal termination") + 1
        n_basis_d = overlap = fock = None
        for i in range(second, len(lines)):
            line = lines[i]
            if n_basis_d is None:
                n_basis_d = Tcal.extract_num("[0-9]+ cartesian basis functions", line)
            if "*** Overlap ***" in line:
                overlap = Tcal.read_symmetric_matrix(io.StringIO("".join(lines[i + 1 :])), n_basis_d)
            if "Fock matrix (alpha):" in line:
                fock = Tcal.read_symmetric_matrix(io.StringIO("".join(lines[i + 1 :])), n_basis_d)
                break
        if overlap is None or fock is None:
            raise ValueError(f"Failed to parse overlap/Fock from {log}")
        return overlap, fock

    @staticmethod
    def _read_log_lines(log: Path) -> list[str]:
        """Read a log file and verify it ended with a normal termination."""
        with open(log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not any("Normal termination" in line for line in lines[-3:]):
            raise RuntimeError(f"{log} did not terminate normally.")
        return lines

    @staticmethod
    def _find(lines: list[str], token: str) -> int:
        """Return the index of the first line containing ``token`` (or -1)."""
        for idx, line in enumerate(lines):
            if token in line:
                return idx
        return -1

    @staticmethod
    def _extract_energy(lines: list[str]) -> np.ndarray:
        """Collect orbital eigenvalues (Hartree) from a Gaussian log."""
        num_orbitals = lumo_index = None
        for line in reversed(lines):
            if num_orbitals is None:
                num_orbitals = Tcal.extract_num("[0-9]+ cartesian basis functions", line)
            if lumo_index is None:
                lumo_index = Tcal.extract_num("[0-9]+ alpha electrons", line)
            if num_orbitals is not None and lumo_index is not None:
                break
        valence_rows = -(-lumo_index // 5)
        conduction_rows = -(-(num_orbitals - lumo_index) // 5)
        eig_lines = [
            line for line in lines
            if "Alpha  occ. eigenvalues" in line or "Alpha virt. eigenvalues" in line
        ][-(valence_rows + conduction_rows):]
        energies = np.empty(num_orbitals)
        count = 0
        for line in eig_lines:
            line = line.strip()
            fields = [line[27:37], line[37:47], line[47:57], line[57:67], line[67:77]]
            vals = [float(v) for v in fields if v.strip()]
            energies[count : count + len(vals)] = vals
            count += len(vals)
        return energies

    @staticmethod
    def _read_orbital_str(lines: list[str], start_index: int, n_basis: int) -> list[str]:
        """Read cartesian AO type labels from a ``Molecular Orbital Coefficients`` block."""
        orbital_str = []
        read_start = lines[start_index + 4].index("1S")
        for i in range(start_index + 4, start_index + 4 + n_basis):
            orbital_str.append(lines[i][read_start - 1 : read_start + 3].strip())
        return orbital_str

    @staticmethod
    def _orbital_str_to_ao_meta(orbital_str: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Gaussian cartesian AO labels to ``(ao_atom, ao_xyz)``.

        ``ao_atom[k]`` is the 0-based atom index of AO ``k`` (atoms start at a
        ``"1S"`` label); ``ao_xyz[k]`` are the cartesian powers ``(nx, ny, nz)``
        obtained by counting x/y/z characters in the label.
        """
        nb = len(orbital_str)
        ao_xyz = np.array(
            [[s.lower().count("x"), s.lower().count("y"), s.lower().count("z")] for s in orbital_str],
            dtype=np.int64,
        )
        starts = [k for k, s in enumerate(orbital_str) if s.upper() == "1S"]
        ao_atom = np.zeros(nb, dtype=np.int64)
        bounds = starts + [nb]
        for atom, (st, en) in enumerate(zip(bounds[:-1], bounds[1:])):
            ao_atom[st:en] = atom
        return ao_atom, ao_xyz

    # ------------------------------------------------------------------ #
    # PySCF / gpu4pyscf engine                                           #
    # ------------------------------------------------------------------ #
    def _run_dft_pyscf(self, run_scf: bool, resume: bool) -> None:
        """Run/parse PySCF for all monomers (skip=[2,3]) and dimers (skip=[1,2]).

        Each unique monomer and dimer SCF is executed exactly once
        (``m + D`` total). Cartesian basis is forced for symmetry consistency.
        """
        from pyscf import lib

        from tcal import TcalPySCF

        use_gpu = self.engine == "gpu4pyscf"

        def make(stem: str, mono1_atoms: int) -> TcalPySCF:
            xyz = self.logs_dir / f"{stem}.xyz"
            shutil.copy(self.inputs_dir / f"{stem}.xyz", xyz)
            return TcalPySCF(
                str(xyz),
                monomer1_atom_num=mono1_atoms,
                method=self.method,
                use_gpu=use_gpu,
                ncore=self.cpu,
                max_memory_gb=self.mem,
                cart=True,
                bse=self.bse,
            )

        n_elect_arr = np.empty(self.m, dtype=np.int64)
        for pos, site in enumerate(self.sites, start=1):
            a = int(site["idx"])
            stem = f"{self.name}_monomer_{a:03d}"
            tcal = make(stem, self.n_atoms)
            chk = self.logs_dir / f"{stem}_m1.chk"
            skipped = run_scf and resume and self._pyscf_completed(chk)
            if not skipped and run_scf:
                self._begin_job("monomer", pos, self.m, stem)
                tcal.run_pyscf(skip_monomer_num=[2, 3], verbose=False)
            tcal.read_monomer1()  # prints its own "reading ..._m1.chk"
            mo = tcal.mo1  # (n_basis, n_bsuse) = (AO, MO)
            if tcal._mf1 is not None:
                ene = tcal._to_numpy(tcal._mf1.mo_energy)
                mol = tcal._mf1.mol
            else:
                ene = np.asarray(lib.chkfile.load(str(chk), "scf/mo_energy"))
                mol = lib.chkfile.load_mol(str(chk))
            if skipped:
                self._skip_job("monomer", pos, self.m, stem)
            else:
                ok = bool(tcal._mf1.converged) if tcal._mf1 is not None else self._pyscf_completed(chk)
                self._finish_job("monomer", pos, self.m, stem, ok, fail_status="NOT CONVERGED")
            if self.MOs is None:
                nb = mo.shape[0]
                self.MOs = np.empty((self.m, nb, mo.shape[1]))
                self.energies = np.empty((self.m, ene.shape[0]))
                self.ao_atom, self.ao_xyz = self._pyscf_ao_meta(mol)
                self.n_basis = nb
            self.MOs[a] = mo
            self.energies[a] = ene * HARTREE_TO_EV
            n_elect_arr[a] = int(tcal.n_elect1)

        for pos, dt in enumerate(self.dimer_types, start=1):
            i = int(dt["ti_flg"])
            stem = f"{self.name}_dimer_{i:03d}"
            n_atoms_cent = len(self.sites[int(dt["sites"][0])]["symbols"])
            tcal = make(stem, n_atoms_cent)
            chk = self.logs_dir / f"{stem}.chk"
            skipped = run_scf and resume and self._pyscf_completed(chk)
            if not skipped and run_scf:
                self._begin_job("dimer", pos, self.num_dimer_types, stem)
                tcal.run_pyscf(skip_monomer_num=[1, 2], verbose=False)
            self._reading(chk)
            mol, fock = tcal._load_from_chk_dimer(chkfile=str(chk), mf=tcal._mf_d)
            ov = np.asarray(mol.intor("int1e_ovlp"))
            fk = tcal._to_numpy(fock)
            if skipped:
                self._skip_job("dimer", pos, self.num_dimer_types, stem)
            else:
                ok = bool(tcal._mf_d.converged) if tcal._mf_d is not None else self._pyscf_completed(chk)
                self._finish_job("dimer", pos, self.num_dimer_types, stem, ok, fail_status="NOT CONVERGED")
            if self.overlap_ori is None:
                nbd = ov.shape[0]
                self.overlap_ori = np.empty((self.num_dimer_types, nbd, nbd))
                self.fock_ori = np.empty((self.num_dimer_types, nbd, nbd))
                self.n_basis_d = nbd
            self.overlap_ori[i] = ov
            self.fock_ori[i] = fk

        if not (n_elect_arr == n_elect_arr[0]).all():
            raise ValueError(f"Inconsistent electron counts across monomers: {n_elect_arr}")
        self.n_elect = int(n_elect_arr[0])

    @staticmethod
    def _pyscf_ao_meta(mol) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(ao_atom, ao_xyz)`` from a cartesian PySCF ``Mole``."""
        ao_atom = []
        ao_xyz = []
        for iatom, _sym, _nl, ml in mol.ao_labels(fmt=False):
            ao_atom.append(int(iatom))
            ml = ml.lower()
            ao_xyz.append([ml.count("x"), ml.count("y"), ml.count("z")])
        return np.array(ao_atom, dtype=np.int64), np.array(ao_xyz, dtype=np.int64)

    @staticmethod
    def _pyscf_completed(chk: Path) -> bool:
        """Return True if the PySCF chkfile records a converged, completed SCF.

        The ``job_status/completed`` flag is written by ``TcalPySCF`` only when
        the SCF converged, so it distinguishes a finished job from a chkfile
        left behind by an interrupted or unconverged run.

        Parameters
        ----------
        chk : pathlib.Path
            Path to the PySCF checkpoint file.

        Returns
        -------
        bool
            ``True`` if the chkfile exists and marks the SCF as completed.
        """
        if not chk.exists():
            return False
        from pyscf import lib

        try:
            return bool(lib.chkfile.load(str(chk), "job_status/completed"))
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # ORCA engine                                                        #
    # ------------------------------------------------------------------ #
    def _run_dft_orca(self, run_scf: bool, resume: bool) -> None:
        """Run/parse ORCA for all monomers (skip=[2,3]) and dimers (skip=[1,2])."""
        from tcal import TcalORCA

        def make(stem: str, mono1_atoms: int) -> TcalORCA:
            xyz = self.logs_dir / f"{stem}.xyz"
            shutil.copy(self.inputs_dir / f"{stem}.xyz", xyz)
            return TcalORCA(
                str(xyz),
                monomer1_atom_num=mono1_atoms,
                method=self.method,
                ncore=self.cpu,
                max_memory_mb=self.mem * 1024,
            )

        n_elect_arr = np.empty(self.m, dtype=np.int64)
        for pos, site in enumerate(self.sites, start=1):
            a = int(site["idx"])
            stem = f"{self.name}_monomer_{a:03d}"
            tcal = make(stem, self.n_atoms)
            skipped = run_scf and resume and self._orca_completed(tcal, f"{stem}_m1", "_output1")
            if not skipped and run_scf:
                self._begin_job("monomer", pos, self.m, stem)
                tcal.run_orca(skip_monomer_num=[2, 3], verbose=False)
            tcal.read_monomer1()  # prints its own "reading ..._m1.out"
            # ORCA stores mo as (MO, AO); transpose to (AO, MO).
            mo = np.asarray(tcal.mo1).T
            output = tcal._output1 or tcal._load_output(
                f"{stem}_m1", self.logs_dir.resolve()
            )
            mos = output.get_mos()["mo"]
            ene = np.array([orb.orbitalenergy for orb in mos])
            if skipped:
                self._skip_job("monomer", pos, self.m, stem)
            else:
                ok = bool(output.terminated_normally() and output.scf_converged())
                self._finish_job("monomer", pos, self.m, stem, ok)
            if self.MOs is None:
                nb = mo.shape[0]
                self.MOs = np.empty((self.m, nb, mo.shape[1]))
                self.energies = np.empty((self.m, ene.shape[0]))
                self.ao_atom, self.ao_xyz = self._orca_ao_meta(tcal)
                self.n_basis = nb
            self.MOs[a] = mo
            self.energies[a] = ene * HARTREE_TO_EV
            n_elect_arr[a] = int(tcal.n_elect1)

        for pos, dt in enumerate(self.dimer_types, start=1):
            i = int(dt["ti_flg"])
            stem = f"{self.name}_dimer_{i:03d}"
            n_atoms_cent = len(self.sites[int(dt["sites"][0])]["symbols"])
            tcal = make(stem, n_atoms_cent)
            skipped = run_scf and resume and self._orca_completed(tcal, stem, "_output_d")
            if not skipped and run_scf:
                self._begin_job("dimer", pos, self.num_dimer_types, stem)
                tcal.run_orca(skip_monomer_num=[1, 2], verbose=False)
            # Reuse TcalORCA.read_dimer matrix extraction; it needs monomer
            # sizes for the (unused) MO zero-pad, so provide them.
            tcal.n_bsuse1 = tcal.n_basis1 = self.n_basis
            tcal.n_bsuse2 = tcal.n_basis2 = self.n_basis
            tcal.mo1 = np.zeros((self.n_basis, self.n_basis))
            tcal.mo2 = np.zeros((self.n_basis, self.n_basis))
            tcal.read_dimer()  # prints its own "reading ....out"
            ov = np.asarray(tcal.overlap)
            fk = np.asarray(tcal.fock)
            if skipped:
                self._skip_job("dimer", pos, self.num_dimer_types, stem)
            else:
                output_d = tcal._output_d or tcal._load_output(stem, self.logs_dir.resolve())
                ok = bool(output_d.terminated_normally() and output_d.scf_converged())
                self._finish_job("dimer", pos, self.num_dimer_types, stem, ok)
            if self.overlap_ori is None:
                nbd = ov.shape[0]
                self.overlap_ori = np.empty((self.num_dimer_types, nbd, nbd))
                self.fock_ori = np.empty((self.num_dimer_types, nbd, nbd))
                self.n_basis_d = nbd
            self.overlap_ori[i] = ov
            self.fock_ori[i] = fk

        if not (n_elect_arr == n_elect_arr[0]).all():
            raise ValueError(f"Inconsistent electron counts across monomers: {n_elect_arr}")
        self.n_elect = int(n_elect_arr[0])

    @staticmethod
    def _orca_label_powers(orbital: str) -> list[int]:
        """Per-axis exponent sums of an ORCA real-spherical-harmonic label.

        ORCA labels real solid harmonics with cartesian-monomial names that
        carry exponents (``s``, ``pz``, ``dz2``, ``dx2y2``, ``dxy`` ...). Each
        such harmonic has a definite parity in every axis, and summing the
        exponent digit following each ``x``/``y``/``z`` letter (1 if absent)
        reproduces it. Counting bare letters instead (as a naive parser does)
        misreads ``dz2`` as odd in ``z`` and ``dx2y2`` as odd in ``x``/``y``,
        corrupting the axis-flip sign factors used by the crystal-symmetry
        embedding. Only the parity (even/odd) is consumed downstream via
        ``s ** ao_xyz`` with ``s`` in ``{+1, -1}``.

        Parameters
        ----------
        orbital : str
            ORCA orbital-type label, e.g. ``"1dz2"`` or ``"2pz"``.

        Returns
        -------
        list[int]
            ``[lx, ly, lz]`` exponent sums per axis.

        Examples
        --------
        >>> Bcal._orca_label_powers("1dz2")
        [0, 0, 2]
        >>> Bcal._orca_label_powers("1dx2y2")
        [2, 2, 0]
        >>> Bcal._orca_label_powers("1dxy")
        [1, 1, 0]
        """
        powers = {"x": 0, "y": 0, "z": 0}
        for axis, digits in re.findall(r"([xyz])(\d*)", orbital.lower()):
            powers[axis] += int(digits) if digits else 1
        return [powers["x"], powers["y"], powers["z"]]

    @staticmethod
    def _orca_ao_meta(tcal) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(ao_atom, ao_xyz)`` from ORCA monomer AO labels."""
        output = tcal._output1
        output.recreate_gbw_results({})
        tcal._build_ao_labels_from_orca(output)
        ao_atom = np.array(tcal.atom_index, dtype=np.int64)
        ao_xyz = np.array(
            [Bcal._orca_label_powers(o) for o in tcal.atom_orbital],
            dtype=np.int64,
        )
        return ao_atom, ao_xyz

    def _orca_completed(self, tcal, basename: str, slot: str) -> bool:
        """Return True if an ORCA job finished normally with a converged SCF.

        The parsed OPI ``Output`` is cached onto ``tcal`` (attribute ``slot``)
        so the subsequent ``read_monomer1``/``read_dimer`` reuses it instead of
        re-parsing the output from disk.

        Parameters
        ----------
        tcal : TcalORCA
            The ORCA helper bound to this job (provides ``_load_output``).
        basename : str
            ORCA job basename in ``logs/`` (e.g. ``"<name>_monomer_000_m1"``).
        slot : str
            ``tcal`` attribute to cache the parsed ``Output`` into
            (``"_output1"`` for a monomer, ``"_output_d"`` for a dimer).

        Returns
        -------
        bool
            ``True`` if the ``.out`` exists, terminated normally, and converged.
        """
        out = self.logs_dir / f"{basename}.out"
        if not out.exists():
            return False
        try:
            output = tcal._load_output(basename, self.logs_dir.resolve())
            if not (output.terminated_normally() and output.scf_converged()):
                return False
        except Exception:
            return False
        setattr(tcal, slot, output)
        return True

    # ------------------------------------------------------------------ #
    # Transfer integrals (crystal symmetry)                             #
    # ------------------------------------------------------------------ #
    def cal_transfer_integrals_all(self) -> np.ndarray:
        """Build the transfer-integral tensor for all pairs using symmetry.

        Returns
        -------
        numpy.ndarray, shape (n_pairs, n, n)
            Transfer integrals in eV for the ``n``-orbital window of each pair.
        """
        focks = self.fock_ori[self.pairs[:, 5]].copy()
        overlaps = self.overlap_ori[self.pairs[:, 5]].copy()
        h = self.overlap_ori.shape[-1]
        inv_index = list(range(h // 2, h)) + list(range(h // 2))
        transposed = self.pairs[:, 6] == 1
        focks[transposed] = focks[transposed][:, inv_index][:, :, inv_index]
        overlaps[transposed] = overlaps[transposed][:, inv_index][:, :, inv_index]

        mo1, clean1 = self._build_embedded_mo(self.pairs[:, 0], first_block=True, mat=overlaps)
        mo2, clean2 = self._build_embedded_mo(self.pairs[:, 1], first_block=False, mat=overlaps)

        self.trans = self._cal_transfer(mo1, overlaps, focks, mo2) * 1e-3  # meV -> eV
        self._enforce_hermiticity(clean1 & clean2)
        return self.trans

    def _enforce_hermiticity(self, clean: np.ndarray) -> None:
        """Make reverse-pair transfer blocks exact transposes of forward ones.

        Hermiticity of the tight-binding ``H(k)`` requires the block of pair
        ``(b, a, -R)`` to equal the transpose of the block of pair ``(a, b, R)``
        (the dimer Fock matrix is symmetric, so ``<a_x|F|b_y> = <b_y|F|a_x>``).
        The per-pair self-overlap repair in :meth:`_build_embedded_mo` restores
        the self-overlap but may pick a monomer automorphism that differs
        between a pair and its reverse, breaking that transpose relation and
        making ``H(k)`` non-Hermitian. Here the trusted member of each
        ``{(a, b, R), (b, a, -R)}`` orbit -- the one embedded with the identity
        automorphism (``clean``), which preserves the monomer orbital gauge --
        is propagated to its partner by transposition.

        Parameters
        ----------
        clean : numpy.ndarray of bool, shape (n_pairs,)
            True where both monomers of the pair used the identity automorphism.
        """
        pairs = self.pairs[:, :5]
        index = {tuple(int(v) for v in row): i for i, row in enumerate(pairs)}
        unresolved = False
        orphans: list[tuple[int, int, int, int, int]] = []
        for i, (cent, nei, sx, sy, sz) in enumerate(pairs):
            partner = index.get((int(nei), int(cent), -int(sx), -int(sy), -int(sz)))
            if partner is None:
                # No reverse pair (b, a, -R) for this (a, b, R): the block has
                # no transpose to be matched against, so H(k) cannot be made
                # exactly Hermitian. This points to a pair dropped during input
                # generation (see compare_coordinates in input_maker.py).
                orphans.append((int(cent), int(nei), int(sx), int(sy), int(sz)))
                continue
            if partner == i:
                # Self-reverse pair (on-site, R = 0): force the block symmetric.
                self.trans[i] = 0.5 * (self.trans[i] + self.trans[i].T)
            elif clean[partner] and not clean[i]:
                self.trans[i] = self.trans[partner].T
            elif not clean[i] and not clean[partner]:
                # Neither member is trustworthy: symmetrise to stay Hermitian.
                herm = 0.5 * (self.trans[i] + self.trans[partner].T)
                self.trans[i] = herm
                self.trans[partner] = herm.T
                unresolved = True
        if unresolved:
            logger.warning("Transfer-integral gauge unresolved for some pairs; symmetrised H(k).")
        if orphans:
            logger.warning(
                f"{len(orphans)} pair(s) have no reverse partner; "
                f"H(k) cannot be made Hermitian: {orphans}"
            )

    def _build_embedded_mo(
        self, mol_idx: np.ndarray, first_block: bool, mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed monomer MOs into the dimer basis with symmetry-correct alignment.

        For each pair the monomer MO is transformed (window slice, zero-pad,
        atom reorder, AO sign flip) into the dimer basis. The geometric
        ``atoms_order`` can be ambiguous for molecules with point-group symmetry
        (it may pick a self-symmetry image), so among the monomer's own
        automorphisms the one giving unit self-overlap is selected per pair.

        Parameters
        ----------
        mol_idx : numpy.ndarray, shape (n_pairs,)
            Site index of the molecule (molecule 1 or 2) for each pair.
        first_block : bool
            If True, the molecule occupies the first AO block (molecule 1);
            otherwise the second block (molecule 2).
        mat : numpy.ndarray, shape (n_pairs, 2*n_basis, 2*n_basis)
            Per-pair dimer overlap matrix used for the self-overlap check.

        Returns
        -------
        mo : numpy.ndarray, shape (n_pairs, n, 2*n_basis)
            Embedded MO coefficient vectors.
        used_identity : numpy.ndarray of bool, shape (n_pairs,)
            True for pairs where the identity automorphism was kept (i.e. no
            self-symmetry repair was applied). Such embeddings preserve the
            monomer orbital gauge and are the trusted member of each
            reverse-pair orbit; see :meth:`_enforce_hermiticity`.
        """
        n = self.n
        nb = self.n_basis
        start, end = -(n // 2), n // 2
        sign_conv = np.prod(
            self.pairs[:, 7:10].astype(np.float64)[:, np.newaxis, :] ** self.ao_xyz[np.newaxis, :, :], axis=2
        )
        candidates = self._automorphism_candidates()  # identity is always candidate 0

        best_mo: Optional[np.ndarray] = None
        best_dev: Optional[np.ndarray] = None
        best_idx: Optional[np.ndarray] = None
        for cand_idx, (ao_src, ao_sign) in enumerate(candidates):
            # Apply the candidate automorphism to every monomer MO on the AO axis.
            mo = self.MOs[mol_idx][:, ao_src, :] * ao_sign[np.newaxis, :, np.newaxis]
            mo = mo.transpose(0, 2, 1)[:, self.n_elect + start : self.n_elect + end, :]
            if first_block:
                mo = np.concatenate([mo, np.zeros_like(mo)], axis=-1)
            else:
                mo = np.concatenate([np.zeros_like(mo), mo], axis=-1)
            self._apply_atom_reorder(mo)
            if first_block:
                mo[:, :, :nb] *= sign_conv[:, np.newaxis, :]
                dev = np.abs(_self_term_bra(mo, mat)[:, :, 0] - 1).max(axis=1)
            else:
                mo[:, :, nb:] *= sign_conv[:, np.newaxis, :]
                dev = np.abs(_self_term_ket(mo, mat)[:, 0, :] - 1).max(axis=1)

            if best_mo is None:
                best_mo, best_dev = mo, dev
                best_idx = np.zeros(dev.shape[0], dtype=np.int64)
            else:
                # Only repair pairs that still fail the self-overlap check; the
                # identity (first candidate) is kept for already-correct pairs,
                # since applying an automorphism there would keep S=1 but flip
                # the parity/sign and corrupt the (otherwise correct) integral.
                improve = (dev < best_dev) & (best_dev > 1e-3)
                best_mo[improve] = mo[improve]
                best_dev[improve] = dev[improve]
                best_idx[improve] = cand_idx

        if best_dev.max() > 1e-2:
            logger.warning(f"Self overlap is not unity: max |S-1| = {best_dev.max():.3e}")
        return best_mo, best_idx == 0

    def _automorphism_candidates(self) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Return the de-duplicated union of all sites' monomer automorphisms.

        The identity is always first. Candidates are used to resolve the
        atom-correspondence ambiguity of symmetric molecules by selecting, per
        pair, the one giving unit self-overlap.

        Returns
        -------
        list of (numpy.ndarray, numpy.ndarray)
            ``(ao_src, ao_sign)`` AO-level operations.
        """
        if "_union" in self._autos_cache:
            return self._autos_cache["_union"]
        candidates: list[Tuple[np.ndarray, np.ndarray]] = []
        seen: set = set()
        for site in range(self.m):
            for ao_src, ao_sign in self._monomer_automorphisms(site):
                key = (ao_src.tobytes(), ao_sign.tobytes())
                if key not in seen:
                    seen.add(key)
                    candidates.append((ao_src, ao_sign))
        self._autos_cache["_union"] = candidates
        return candidates

    def _apply_atom_reorder(self, mo: np.ndarray) -> None:
        """Reorder MO AO columns in place per pair according to ``atoms_order``."""
        if self.atoms_order is None:
            return
        nb = self.n_basis
        n = self.n
        ranges = self._atom_ao_ranges()
        for unique_ao in np.unique(self.atoms_order, axis=0):
            mo_order = np.concatenate([np.arange(ranges[a, 0], ranges[a, 1]) for a in unique_ao])
            half = mo_order.shape[0] // 2
            mo_order = mo_order.copy()
            mo_order[half:] += nb
            mask = (self.atoms_order == unique_ao).all(axis=-1)
            idx = np.where(mask)[0]
            mo[mask] = mo[np.ix_(idx, np.arange(n), mo_order)]

    def _monomer_automorphisms(self, site_idx: int) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Geometric self-symmetries of a monomer as AO-level (perm, sign) ops.

        A monomer automorphism is an axis sign-flip ``s`` (in ``{+1,-1}^3``)
        together with the atom permutation that maps the (centered) molecule
        onto itself. Identity is always included. These are used to resolve the
        atom-correspondence ambiguity of symmetric molecules.

        Parameters
        ----------
        site_idx : int
            Index of the unique molecule.

        Returns
        -------
        list of (numpy.ndarray, numpy.ndarray)
            For each automorphism, ``(ao_src, ao_sign)``: the AO source-gather
            indices (length n_basis) and the per-AO sign factor.
        """
        if site_idx in self._autos_cache:
            return self._autos_cache[site_idx]
        cart = np.asarray(self.sites[site_idx]["frac"]) @ self.lattice
        cart = cart - cart.mean(axis=0)
        symbols = np.asarray(self.sites[site_idx]["symbols"])
        ranges = self._atom_ao_ranges()
        block_sizes = ranges[:, 1] - ranges[:, 0]
        autos: list[Tuple[np.ndarray, np.ndarray]] = []
        for s in itertools.product((1, -1), repeat=3):
            s = np.array(s)
            transformed = cart * s
            dist = np.sqrt(((transformed[:, np.newaxis, :] - cart[np.newaxis, :, :]) ** 2).sum(axis=-1))
            perm = dist.argmin(axis=1)
            # A genuine automorphism is a bijection that maps each atom onto a
            # coincident one of the same element (hence the same AO-block size).
            if np.unique(perm).size != self.n_atoms or dist[np.arange(self.n_atoms), perm].max() >= 1e-3:
                continue
            if not (symbols[perm] == symbols).all() or not (block_sizes[perm] == block_sizes).all():
                continue
            ao_src = np.empty(self.n_basis, dtype=np.int64)
            for a in range(self.n_atoms):
                s0, e0 = ranges[a]
                s1, e1 = ranges[perm[a]]
                ao_src[s0:e0] = np.arange(s1, e1)
            ao_sign = np.prod(s.astype(np.float64) ** self.ao_xyz, axis=1)
            autos.append((ao_src, ao_sign))
        self._autos_cache[site_idx] = autos
        return autos

    def _atom_ao_ranges(self) -> np.ndarray:
        """Return per-atom AO ``[start, end)`` ranges from ``ao_atom``."""
        ranges = np.zeros((self.n_atoms, 2), dtype=np.int64)
        for atom in range(self.n_atoms):
            idx = np.where(self.ao_atom == atom)[0]
            ranges[atom] = [idx[0], idx[-1] + 1]
        return ranges

    def _cal_transfer(
        self, bra: np.ndarray, overlap: np.ndarray, fock: np.ndarray, ket: np.ndarray
    ) -> np.ndarray:
        """Vectorized fragment-MO transfer integrals (same formula as ``Tcal``).

        Parameters
        ----------
        bra, ket : numpy.ndarray, shape (n_pairs, n, 2*n_basis)
            MO coefficient vectors of molecule 1 / 2 in the dimer basis.
        overlap, fock : numpy.ndarray, shape (n_pairs, 2*n_basis, 2*n_basis)
            Dimer overlap / Fock matrices.

        Returns
        -------
        numpy.ndarray, shape (n_pairs, n, n)
            Transfer integrals in meV.
        """
        s11 = _self_term_bra(bra, overlap)
        s22 = _self_term_ket(ket, overlap)
        s12 = bra @ overlap @ ket.transpose(0, 2, 1)
        f11 = _self_term_bra(bra, fock)
        f22 = _self_term_ket(ket, fock)
        f12 = bra @ fock @ ket.transpose(0, 2, 1)
        if (np.abs(s11 - 1) > 1e-2).any():
            logger.warning(f"Self overlap is not unity: max |S11-1| = {np.abs(s11 - 1).max():.3e}")
        if (np.abs(s22 - 1) > 1e-2).any():
            logger.warning(f"Self overlap is not unity: max |S22-1| = {np.abs(s22 - 1).max():.3e}")
        return ((f12 - 0.5 * (f11 + f22) * s12) / (1 - s12 * s12)) * self.EV

    # ------------------------------------------------------------------ #
    # Tight-binding band structure & effective mass                     #
    # ------------------------------------------------------------------ #
    def cal_reciprocal_lattice(self) -> np.ndarray:
        """Calculate reciprocal lattice vectors from the direct lattice."""
        numerator = np.cross(self.lattice[[1, 2, 0], :], self.lattice[[2, 0, 1], :])
        denominator = self.lattice[0] @ np.cross(self.lattice[1], self.lattice[2])
        return 2 * np.pi * numerator / denominator

    def energies_at_ks(self, k: np.ndarray) -> np.ndarray:
        """Calculate band energies at given k-points.

        Parameters
        ----------
        k : numpy.ndarray, shape (N, 3) or (3,)
            k-point vectors in reciprocal space.

        Returns
        -------
        numpy.ndarray, shape (N, m*n)
            Band energies (eV) at each k-point.
        """
        m, n = self.m, self.n
        lattice = self.lattice
        pairs = self.pairs[:, :5]
        trans = self.trans
        levels = self.levels

        if k.ndim == 1:
            k = k.reshape(1, 3)

        h = np.zeros((k.shape[0], m, n, m, n), dtype=np.complex128)
        for i in range(m):
            for j in range(n):
                h[:, i, j, i, j] += levels[i, j]

        phase_trans = (
            np.exp(
                k.reshape(k.shape[0], 1, 1, 3)
                @ (pairs[:, 2:5].reshape(-1, 1, 3) @ lattice).transpose(0, 2, 1) * 1j
            )
            .reshape(k.shape[0], pairs.shape[0], 1, 1)
            .repeat(n, axis=2)
            .repeat(n, axis=3)
            * trans.reshape(1, trans.shape[0], trans.shape[1], trans.shape[2]).repeat(k.shape[0], axis=0)
        ).transpose(1, 0, 2, 3)
        h = h.transpose(1, 3, 0, 2, 4)
        np.add.at(h, (pairs[:, 0], pairs[:, 1]), phase_trans)
        h = h.transpose(2, 0, 3, 1, 4)
        if not self._hermitian_warned and not _is_hermitian(h.reshape(k.shape[0], m * n, m * n)):
            logger.warning("Hamiltonian is not Hermitian matrix.")
            self._hermitian_warned = True

        return LA.eigvalsh(h.reshape(k.shape[0], m * n, m * n))

    def cal_effective_mass(self, draw_band: bool = True, band_path: str = "XGYGZ") -> np.ndarray:
        """Calculate effective masses at the HOMO maximum and LUMO minimum.

        Parameters
        ----------
        draw_band : bool, optional
            If True, draw and save a band diagram, by default True.
        band_path : str, optional
            High-symmetry k-path passed to :meth:`draw_band_diagram` when
            ``draw_band`` is True, by default ``"XGYGZ"``.

        Returns
        -------
        numpy.ndarray, shape (6,)
            Principal effective masses (in units of electron mass): the first
            three at the HOMO maximum, the last three at the LUMO minimum.

        Notes
        -----
        The band-edge k-points are also stored on the instance as
        ``homo_edge_frac``/``lumo_edge_frac`` (fractional reciprocal-lattice
        coordinates) and ``homo_edge_label``/``lumo_edge_label`` (the matching
        high-symmetry label from :data:`HIGH_SYMMETRY_POINTS`, or ``None``).

        The principal effective masses (sorted by ``|m|`` ascending) and their
        unit principal-axis vectors (rows, Cartesian reciprocal-space) are
        stored as ``homo_em``/``lumo_em`` and
        ``homo_em_vectors``/``lumo_em_vectors``; the returned ``(6,)`` array
        keeps the original unsorted layout for backward compatibility.
        """
        rec_lattice = self.cal_reciprocal_lattice()
        rec_lattice_length = np.sqrt(np.sum(rec_lattice ** 2, axis=1))
        split_num = rec_lattice_length // 0.05
        split_num[(split_num % 2 == 0)] += 1
        split_num = split_num.astype(np.int32)
        k_point_0 = np.linspace(-rec_lattice[0] / 2, rec_lattice[0] / 2, split_num[0])
        k_point_1 = np.linspace(-rec_lattice[1] / 2, rec_lattice[1] / 2, split_num[1])
        k_point_2 = np.linspace(-rec_lattice[2] / 2, rec_lattice[2] / 2, split_num[2])
        k_points = np.array(list(itertools.product(k_point_0, k_point_1, k_point_2)))
        homo_ind = int(self.m * self.n / 2 - 1)
        lumo_ind = int(self.m * self.n / 2)

        self.band_energies = self.energies_at_ks(np.sum(k_points, axis=1))
        homo_max_ind = self.band_energies[:, homo_ind].argmax()
        lumo_min_ind = self.band_energies[:, lumo_ind].argmin()
        homo_max_k = np.sum(k_points[homo_max_ind], axis=0)
        lumo_min_k = np.sum(k_points[lumo_min_ind], axis=0)

        homo_max_k = self.refine_around_min_or_max(homo_max_k, homo_ind, max_or_min="max").reshape(1, 3)
        lumo_min_k = self.refine_around_min_or_max(lumo_min_k, lumo_ind, max_or_min="min").reshape(1, 3)

        # Convert the Cartesian band-edge k-points to fractional reciprocal-
        # lattice coordinates (k = f @ rec_lattice, so f = k @ inv(rec_lattice))
        # reduced to (-0.5, 0.5], and label them if they hit a high-symmetry point.
        rec_inv = LA.inv(rec_lattice)
        self.homo_edge_frac = ((homo_max_k @ rec_inv).reshape(3) + 0.5) % 1.0 - 0.5
        self.lumo_edge_frac = ((lumo_min_k @ rec_inv).reshape(3) + 0.5) % 1.0 - 0.5
        self.homo_edge_label = self._kpoint_label(self.homo_edge_frac)
        self.lumo_edge_label = self._kpoint_label(self.lumo_edge_frac)

        eps = 1e-5
        eps_list = [-eps, 0, eps]
        eps_comb = np.array(list(itertools.product(eps_list, eps_list, eps_list)))
        eps_comb[((np.abs(eps_comb).sum(axis=-1) == eps).reshape(-1, 1).repeat(3, axis=-1)) & (eps_comb != 0)] *= 2

        around_homo_max_k = homo_max_k + eps_comb
        around_lumo_min_k = lumo_min_k + eps_comb

        around_homo_ene = self.energies_at_ks(around_homo_max_k)[:, homo_ind].reshape(3, 3, 3)
        around_lumo_ene = self.energies_at_ks(around_lumo_min_k)[:, lumo_ind].reshape(3, 3, 3)

        hbar = 1.054589e-34
        me = 9.109534e-31
        ele = 1.602e-19
        prefactor = (me * ele * 1e-10 ** 2) / hbar ** 2

        homo_em_tensor = prefactor * self._em_tensor(around_homo_ene, eps)
        lumo_em_tensor = prefactor * self._em_tensor(around_lumo_ene, eps)

        homo_em_vals, homo_em_vecs = LA.eig(LA.inv(homo_em_tensor))
        lumo_em_vals, lumo_em_vecs = LA.eig(LA.inv(lumo_em_tensor))
        self.homo_em, self.homo_em_vectors = _sort_em(homo_em_vals, homo_em_vecs)
        self.lumo_em, self.lumo_em_vectors = _sort_em(lumo_em_vals, lumo_em_vecs)
        effective_mass = np.concatenate([homo_em_vals, lumo_em_vals])

        if draw_band:
            self.draw_band_diagram(band_path)
        self._save_results()

        return effective_mass

    @staticmethod
    def _em_tensor(ene: np.ndarray, eps: float) -> np.ndarray:
        """Assemble the 3x3 second-derivative tensor from a 3x3x3 energy stencil."""
        axes = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        return np.array([_second_derivative(ene, eps, ax) for ax in axes]).reshape(3, 3)

    def refine_around_min_or_max(
        self, k_points: np.ndarray, mo_index: int, window_size: float = 0.01, max_or_min: str = "max"
    ) -> np.ndarray:
        """Refine a band extremum k-point with Nelder-Mead optimization."""

        def cal_k_ene(k, mo_index, max_or_min):
            energy = self.energies_at_ks(k.reshape(1, 3))[0, mo_index]
            return energy if max_or_min == "min" else -energy

        return fmin(cal_k_ene, k_points, args=(mo_index, max_or_min), disp=False)

    def draw_band_diagram(self, path: str = "XGYGZ") -> None:
        """Draw and save a band diagram along a high-symmetry k-path.

        The ``path`` string is read one character at a time; each character is a
        high-symmetry point label from :data:`HIGH_SYMMETRY_POINTS` (``"G"``
        denotes Gamma). For ``"XGYGZ"`` the path is X -> Gamma -> Y -> Gamma -> Z.
        The horizontal axis is the cumulative k-distance along the path and the
        ticks are placed at the requested points. The figure is written to
        ``results/band.png`` and the raw data to ``results/band.npz``.

        Parameters
        ----------
        path : str, optional
            Sequence of single-character high-symmetry point labels, by default
            ``"XGYGZ"``.

        Raises
        ------
        ValueError
            If ``path`` contains a label absent from
            :data:`HIGH_SYMMETRY_POINTS`, or has fewer than two points.
        """
        labels = list(path)
        unknown = sorted(set(labels) - set(HIGH_SYMMETRY_POINTS))
        if unknown:
            raise ValueError(
                f"Unknown high-symmetry labels {unknown}; "
                f"available: {' '.join(sorted(HIGH_SYMMETRY_POINTS))}"
            )
        if len(labels) < 2:
            raise ValueError(f"band path must contain at least two points, got {path!r}")

        rec_lattice = self.cal_reciprocal_lattice()
        nodes = np.array([HIGH_SYMMETRY_POINTS[label] for label in labels]) @ rec_lattice

        k_segments: list[np.ndarray] = []
        x_segments: list[np.ndarray] = []
        tick_positions = [0.0]
        running = 0.0
        for start, end in zip(nodes[:-1], nodes[1:]):
            length = float(np.linalg.norm(end - start))
            n_pts = max(int(length // 0.05), 2)
            ratio = np.linspace(0.0, 1.0, n_pts)
            ks = start[np.newaxis, :] * (1 - ratio)[:, np.newaxis] + end[np.newaxis, :] * ratio[:, np.newaxis]
            xs = running + ratio * length
            if k_segments:  # drop the node shared with the previous segment
                ks, xs = ks[1:], xs[1:]
            k_segments.append(ks)
            x_segments.append(xs)
            running += length
            tick_positions.append(running)

        k_path = np.concatenate(k_segments, axis=0)
        x = np.concatenate(x_segments, axis=0)
        band_structure = self.energies_at_ks(k_path).T  # (n_bands, n_k)

        mid = (self.m * self.n) // 2
        upper = min(mid + 10, self.m * self.n - 1)
        lower = max(mid - 11, 0)
        band_max = band_structure[upper].max()
        band_min = band_structure[lower].min()

        plt.rcParams["font.size"] = 18
        fig, ax = plt.subplots(figsize=(9, 16), tight_layout=True)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(band_min - 0.1, band_max + 0.1)
        ax.plot(x, band_structure[: mid - 1].T, color="black")
        ax.plot(x, band_structure[mid + 1 :].T, color="black")
        ax.plot(x, band_structure[mid].T, color="tab:orange")  # LUMO
        ax.plot(x, band_structure[mid - 1].T, color="tab:blue")  # HOMO
        for pos in tick_positions:
            ax.axvline(pos, color="black")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([r"$\Gamma$" if label == "G" else label for label in labels])

        self.results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.results_dir / "band.png", dpi=300)
        plt.close(fig)
        np.savez_compressed(
            self.results_dir / "band.npz",
            band_structure=band_structure,
            x=x,
            tick_positions=np.array(tick_positions),
            labels=np.array(labels),
        )

    def band_edges_on_path(self, path: str) -> dict[str, bool]:
        """Report whether the HOMO/LUMO band edges lie on the band-diagram path.

        A band edge is "on the path" when its k-point lies on one of the
        straight segments connecting the high-symmetry nodes of ``path`` (the
        same geometry drawn by :meth:`draw_band_diagram`), and is therefore
        visible in ``results/band.png``. When an edge is off the path, the
        plotted HOMO/LUMO curve does not reach the true band extremum.

        Parameters
        ----------
        path : str
            High-symmetry k-path string, identical to the one passed to
            :meth:`draw_band_diagram` / :meth:`cal_effective_mass`.

        Returns
        -------
        dict[str, bool]
            Maps ``"HOMO"`` and ``"LUMO"`` to ``True`` if that band edge lies
            on the drawn path.

        Raises
        ------
        RuntimeError
            If the band edges have not been computed yet; call
            :meth:`cal_effective_mass` first.
        """
        if self.homo_edge_frac is None or self.lumo_edge_frac is None:
            raise RuntimeError("Band edges not computed; call cal_effective_mass first.")
        return {
            "HOMO": self._frac_on_band_path(self.homo_edge_frac, path),
            "LUMO": self._frac_on_band_path(self.lumo_edge_frac, path),
        }

    @staticmethod
    def _frac_on_band_path(frac: np.ndarray, path: str, tol: float = 1e-2) -> bool:
        """Return True if a fractional k-point lies on the drawn band path.

        The band path is the sequence of straight segments joining the
        high-symmetry nodes of ``path``. Reciprocal-lattice periodicity is
        handled by testing every periodic image of ``frac`` (translation by an
        integer reciprocal-lattice vector): the point is on the path if any
        image lies within ``tol`` (fractional units) of any segment. The image
        test is required because :meth:`cal_effective_mass` reduces band-edge
        coordinates to ``(-0.5, 0.5]`` (e.g. it stores ``X`` as ``-0.5``, which
        only meets the path's ``+0.5`` node after a ``+1`` translation).

        Parameters
        ----------
        frac : numpy.ndarray, shape (3,)
            Fractional reciprocal-lattice coordinates of the k-point.
        path : str
            Sequence of single-character high-symmetry point labels.
        tol : float, optional
            Distance tolerance in fractional units, by default ``1e-2``.

        Returns
        -------
        bool
            ``True`` if the k-point lies on the band path.
        """
        nodes = np.array([HIGH_SYMMETRY_POINTS[label] for label in path], dtype=np.float64)
        shifts = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
        images = np.asarray(frac, dtype=np.float64) + shifts
        for a, b in zip(nodes[:-1], nodes[1:]):
            d = b - a
            dd = float(d @ d)
            if dd == 0.0:  # degenerate segment (repeated node)
                continue
            t = np.clip((images - a) @ d / dd, 0.0, 1.0)
            closest = a + t[:, np.newaxis] * d
            if np.linalg.norm(images - closest, axis=1).min() < tol:
                return True
        return False

    @staticmethod
    def _kpoint_label(frac: np.ndarray, tol: float = 1e-2) -> Optional[str]:
        """Return the high-symmetry label of a fractional k-point, or ``None``.

        Each fractional component is reduced modulo 1 and snapped to ``0`` or
        ``0.5`` (the only values appearing in :data:`HIGH_SYMMETRY_POINTS`) when
        within ``tol``. If every component snaps, the matching label is returned.

        Parameters
        ----------
        frac : numpy.ndarray, shape (3,)
            Fractional reciprocal-lattice coordinates of the k-point.
        tol : float, optional
            Snap tolerance in fractional units, by default ``1e-2``.

        Returns
        -------
        str or None
            The high-symmetry point label (e.g. ``"U"``), or ``None`` if the
            k-point does not coincide with one.
        """
        snapped: list[float] = []
        for v in np.asarray(frac) % 1.0:
            if abs(v) < tol or abs(v - 1.0) < tol:
                snapped.append(0.0)
            elif abs(v - 0.5) < tol:
                snapped.append(0.5)
            else:
                return None
        return _LABEL_BY_FRAC.get(tuple(snapped))

    def _save_results(self) -> None:
        """Write ``results/effective_mass.csv`` (HOMO/LUMO principal masses).

        Each row records the three principal masses (``em_1``/``em_2``/``em_3``,
        sorted by ``|m|`` ascending), the matching unit principal-axis vectors
        (``v{i}_x``/``v{i}_y``/``v{i}_z``, Cartesian reciprocal-space), the
        band-edge k-point as fractional reciprocal-lattice coordinates
        (``k1``/``k2``/``k3``), and its high-symmetry ``label`` (empty when the
        k-point is not a high-symmetry point).
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                [*self.homo_em, *self.homo_em_vectors.reshape(-1),
                 *self.homo_edge_frac, self.homo_edge_label or ""],
                [*self.lumo_em, *self.lumo_em_vectors.reshape(-1),
                 *self.lumo_edge_frac, self.lumo_edge_label or ""],
            ],
            index=["HOMO", "LUMO"],
            columns=[
                "em_1", "em_2", "em_3",
                "v1_x", "v1_y", "v1_z", "v2_x", "v2_y", "v2_z", "v3_x", "v3_y", "v3_z",
                "k1", "k2", "k3", "label",
            ],
        )
        df.to_csv(self.results_dir / "effective_mass.csv")

    # ------------------------------------------------------------------ #
    # subprocess helper                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _execute(command_list: list[str], cwd: Optional[Path] = None) -> None:
        """Run a subprocess command, logging stderr to ``stderr`` on failure."""
        res = subprocess.run(command_list, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
        if res.returncode:
            logger.error(res.stderr.strip())


def _self_term_bra(bra: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Diagonal ``<psi_i|M|psi_i>`` for molecule 1, shape (n_pairs, n, 1)."""
    if bra.shape[2] % 2 != 0:
        raise ValueError("bra basis dimension must be even (2 * n_basis).")
    out = np.empty((bra.shape[0], bra.shape[1], 1))
    for i in range(bra.shape[1]):
        vec = bra[:, i, :].reshape(bra.shape[0], 1, bra.shape[2])
        out[:, i, :] = (vec @ mat @ vec.transpose(0, 2, 1)).reshape(bra.shape[0], 1)
    return out


def _self_term_ket(ket: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Diagonal ``<psi_j|M|psi_j>`` for molecule 2, shape (n_pairs, 1, n)."""
    if ket.shape[2] % 2 != 0:
        raise ValueError("ket basis dimension must be even (2 * n_basis).")
    out = np.empty((ket.shape[0], 1, ket.shape[1]))
    for i in range(ket.shape[1]):
        vec = ket[:, i, :].reshape(ket.shape[0], 1, ket.shape[2])
        out[:, :, i] = (vec @ mat @ vec.transpose(0, 2, 1)).reshape(ket.shape[0], 1)
    return out


def _sort_em(vals: np.ndarray, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort principal effective masses by ``|m|`` and align their axes.

    Parameters
    ----------
    vals : numpy.ndarray, shape (3,)
        Eigenvalues of the effective-mass tensor (principal masses).
    vecs : numpy.ndarray, shape (3, 3)
        Eigenvectors as columns, ``vecs[:, i]`` paired with ``vals[i]``.

    Returns
    -------
    masses : numpy.ndarray, shape (3,)
        Real principal masses sorted by absolute value in ascending order
        (lightest first); the sign is preserved.
    axes : numpy.ndarray, shape (3, 3)
        Unit principal-axis vectors as rows (Cartesian reciprocal-space),
        ``axes[i]`` paired with ``masses[i]``.
    """
    order = np.argsort(np.abs(vals.real))
    return vals.real[order], vecs.real[:, order].T


def _second_derivative(array: np.ndarray, h: float = 1e-7, axis: str = "xx") -> float:
    """Numerical second-order partial derivative from a 3x3x3 stencil."""
    if axis == "xx":
        return (array[2, 1, 1] - 2 * array[1, 1, 1] + array[0, 1, 1]) / (4 * h ** 2)
    if axis == "yy":
        return (array[1, 2, 1] - 2 * array[1, 1, 1] + array[1, 0, 1]) / (4 * h ** 2)
    if axis == "zz":
        return (array[1, 1, 2] - 2 * array[1, 1, 1] + array[1, 1, 0]) / (4 * h ** 2)
    if axis in ("xy", "yx"):
        return (array[2, 2, 1] + array[0, 0, 1] - array[2, 0, 1] - array[0, 2, 1]) / (4 * h ** 2)
    if axis in ("xz", "zx"):
        return (array[2, 1, 2] + array[0, 1, 0] - array[2, 1, 0] - array[0, 1, 2]) / (4 * h ** 2)
    if axis in ("yz", "zy"):
        return (array[1, 2, 2] + array[1, 0, 0] - array[1, 2, 0] - array[1, 0, 2]) / (4 * h ** 2)
    raise ValueError(f"Unknown axis: {axis}")


def _is_hermitian(array: np.ndarray) -> bool:
    """Check whether a batch of matrices is Hermitian within RMS tolerance."""
    conj = array.conj().transpose(0, 2, 1) if array.ndim == 3 else array.conj().T
    diff = np.sqrt(np.mean(np.abs(array - conj) ** 2)) / np.sqrt(np.mean(np.abs(array) ** 2))
    return bool(np.all(diff[~np.isnan(diff)] <= 1e-6)) if np.ndim(diff) else bool(diff <= 1e-6 or np.isnan(diff))
