"""InputMaker: generate monomer/dimer DFT inputs from a CIF structure.

This module reproduces the monomer/dimer input generation of the legacy
``make_files_for_bcal.py`` (CCDC-based) without the CCDC dependency, by
reusing :class:`mcal.utils.cif_reader.CifReader` and
:class:`mcal.utils.gjf_maker.GjfMaker`.

For each unique molecule in the unit cell a monomer input is written, and for
each symmetry-unique neighboring pair (dimer) a dimer input is written. The
crystal topology (lattice, sites, dimer types, pair table) is saved as a
human-readable ``structure.json``; the bulky per-calculation numerics are left
to later pipeline stages.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mcal.utils.cif_reader import CifReader, FileIO
from mcal.utils.gjf_maker import GjfMaker

from bcal.utils.log import get_logger

logger = get_logger(__name__)


# Atomic weights and van der Waals radii share the same source as CifReader.
ATOMIC_WEIGHTS: Dict[str, float] = CifReader.ATOMIC_WEIGHTS
VAN_DER_WAALS_RADII: Dict[str, float] = (
    CifReader.ELEMENT_PROP[["symbol", "vdw_radius"]]
    .dropna()
    .set_index("symbol")["vdw_radius"]
    .to_dict()
)
# Engines that consume a Gaussian ``.gjf`` input; others receive a plain ``.xyz``.
GAUSSIAN_ENGINES = {"g16", "g09"}


def cal_center_of_weight(
    cart_coos: np.ndarray,
    weight_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate coordinates of center of weight.

    Parameters
    ----------
    cart_coos : np.ndarray, shape (n_atoms, 3)
        Cartesian coordinates in Angstroms.
    weight_array : np.ndarray of shape (n_atoms,), optional
        Atomic weights. If ``None``, the arithmetic mean is returned.

    Returns
    -------
    np.ndarray, shape (3,)
        Center-of-weight coordinates in Angstroms.
    """
    if isinstance(weight_array, np.ndarray):
        center_coos = np.sum(weight_array.reshape(-1, 1) * cart_coos, axis=-2) / np.sum(weight_array)
    else:
        center_coos = np.mean(cart_coos, axis=-2)
    return center_coos


def cal_I(
    coo_array_ori: np.ndarray,
    weight_array_ori: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the inertia tensor and return its eigenvalues and eigenvectors.

    Parameters
    ----------
    coo_array_ori : np.ndarray, shape (n_atoms, 3)
        Cartesian coordinates in Angstroms.
    weight_array_ori : np.ndarray of shape (n_atoms,), optional
        Atomic weights. If ``None``, unit weights are assumed.

    Returns
    -------
    moment : np.ndarray, shape (3,)
        Principal moments of inertia (eigenvalues of the inertia tensor).
    axis : np.ndarray, shape (3, 3)
        Principal axes of inertia (eigenvectors, column-wise).
    """
    coo_array = coo_array_ori.copy()
    if isinstance(weight_array_ori, np.ndarray):
        weight_array = weight_array_ori.copy()
        coo_array -= cal_center_of_weight(coo_array, weight_array)
        I_tensor = np.array([
            [
                np.sum(weight_array * (coo_array[:, 1] ** 2 + coo_array[:, 2] ** 2)),
                np.sum(weight_array * -coo_array[:, 0] * coo_array[:, 1]),
                np.sum(weight_array * -coo_array[:, 0] * coo_array[:, 2]),
            ],
            [
                np.sum(weight_array * -coo_array[:, 0] * coo_array[:, 1]),
                np.sum(weight_array * (coo_array[:, 0] ** 2 + coo_array[:, 2] ** 2)),
                np.sum(weight_array * -coo_array[:, 1] * coo_array[:, 2]),
            ],
            [
                np.sum(weight_array * -coo_array[:, 0] * coo_array[:, 2]),
                np.sum(weight_array * -coo_array[:, 1] * coo_array[:, 2]),
                np.sum(weight_array * (coo_array[:, 1] ** 2 + coo_array[:, 0] ** 2)),
            ],
        ])
    else:
        coo_array -= cal_center_of_weight(coo_array)
        I_tensor = np.array([
            [
                np.sum(coo_array[:, 1] ** 2 + coo_array[:, 2] ** 2),
                np.sum(-coo_array[:, 0] * coo_array[:, 1]),
                np.sum(-coo_array[:, 0] * coo_array[:, 2]),
            ],
            [
                np.sum(-coo_array[:, 0] * coo_array[:, 1]),
                np.sum(coo_array[:, 0] ** 2 + coo_array[:, 2] ** 2),
                np.sum(-coo_array[:, 1] * coo_array[:, 2]),
            ],
            [
                np.sum(-coo_array[:, 0] * coo_array[:, 2]),
                np.sum(-coo_array[:, 1] * coo_array[:, 2]),
                np.sum(coo_array[:, 1] ** 2 + coo_array[:, 0] ** 2),
            ],
        ])
    # The inertia tensor is symmetric; eigh guarantees real eigenvalues
    # (eig may return complex with BLAS-dependent tiny imaginary parts).
    moment, axis = np.linalg.eigh(I_tensor)
    return moment, axis


def check_vdw_contact(
    cart1: np.ndarray,
    symbols1: List[str],
    cart2: np.ndarray,
    symbols2: List[str],
    margin: float = 0.7,
) -> bool:
    """Return True if any atom pair from two molecules is within VdW contact.

    Parameters
    ----------
    cart1 : np.ndarray, shape (n1, 3)
        Cartesian coordinates of molecule 1 in Angstroms.
    symbols1 : list of str
        Element symbols for molecule 1.
    cart2 : np.ndarray, shape (n2, 3)
        Cartesian coordinates of molecule 2 in Angstroms.
    symbols2 : list of str
        Element symbols for molecule 2.
    margin : float, optional
        Extra distance (Angstrom) added to the sum of VdW radii when deciding
        contact, by default 0.7.

    Returns
    -------
    bool
        ``True`` if at least one atom pair is within the VdW contact threshold.
    """
    vdw1 = np.array([VAN_DER_WAALS_RADII.get(s, 2.0) for s in symbols1])
    vdw2 = np.array([VAN_DER_WAALS_RADII.get(s, 2.0) for s in symbols2])
    diff = cart1[:, np.newaxis, :] - cart2[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    contact_thresh = vdw1[:, np.newaxis] + vdw2[np.newaxis, :] + margin
    return bool(np.any(dist <= contact_thresh))


def compare_coordinates(
    dimer_coo: np.ndarray,
    ref_coo: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Determine the symmetry relationship between a dimer and a reference dimer.

    Parameters
    ----------
    dimer_coo : np.ndarray, shape (2, n_atoms, 3)
        Cartesian coordinates of the dimer to compare.
    ref_coo : np.ndarray, shape (2, n_atoms, 3)
        Cartesian coordinates of the reference dimer.

    Returns
    -------
    bool_trans_array : np.ndarray or None
        ``[is_transposed, sign_x, sign_y, sign_z]``. ``None`` only if no
        symmetry operation maps the dimer onto the reference (the two are not
        actually the same dimer type). When several operations match equally
        well -- a benign ambiguity for dimers with their own point-group
        symmetry -- the best one is returned; the resulting orbital-gauge
        difference is absorbed downstream by the self-overlap repair in
        :meth:`Bcal._build_embedded_mo` and :meth:`Bcal._enforce_hermiticity`.
    atoms_order : np.ndarray or None
        Per-atom mapping indices of length ``2 * n_atoms``. ``None`` if no atom
        reordering is needed.
    """
    dimer_coo_at_ori = dimer_coo - dimer_coo.reshape(-1, 3).mean(axis=0)
    ref_coo_at_ori = ref_coo - ref_coo.reshape(-1, 3).mean(axis=0)
    symops = np.array(list(itertools.product([1, -1], [1, -1], [1, -1])))
    min_distance = 10000.0
    best_symop = None
    atoms_order = None
    bool_trans = 0

    # First pass: try simple (same atom-order) matching.
    for symop in symops:
        copy_coo = dimer_coo_at_ori * symop
        distance1 = np.sqrt(((ref_coo_at_ori - copy_coo) ** 2).sum(axis=-1)).mean()
        distance2 = np.sqrt(((ref_coo_at_ori - copy_coo[[1, 0]]) ** 2).sum(axis=-1)).mean()
        distance = distance1 if distance1 < distance2 else distance2
        if distance < min_distance:
            min_distance = distance
            best_symop = symop
            bool_trans = 0 if distance1 < distance2 else 1

    if min_distance >= 1e-8:
        # Second pass: try atom-reordering matching.
        dimer_coo_at_ori = dimer_coo - dimer_coo.reshape(-1, 3).mean(axis=0)
        ref_coo_at_ori = ref_coo - ref_coo.reshape(-1, 3).mean(axis=0)
        symops = np.array(list(itertools.product([1, -1], [1, -1], [1, -1])))
        min_distance = 10000.0
        best_symop = None

        for symop in symops:
            temp_distance = ref_coo_at_ori[:, :, np.newaxis, np.newaxis, :] - (
                dimer_coo_at_ori[np.newaxis, np.newaxis, :, :, :] * symop
            )
            distance1 = np.concatenate([
                np.sqrt((temp_distance[0, :, 0, :] ** 2).sum(axis=-1)).min(axis=-1),
                np.sqrt((temp_distance[1, :, 1, :] ** 2).sum(axis=-1)).min(axis=-1),
            ]).mean()
            distance1_atoms_order = np.concatenate([
                np.sqrt((temp_distance[0, :, 0, :] ** 2).sum(axis=-1)).argmin(axis=-1),
                np.sqrt((temp_distance[1, :, 1, :] ** 2).sum(axis=-1)).argmin(axis=-1),
            ])
            distance2 = np.concatenate([
                np.sqrt((temp_distance[0, :, 1, :] ** 2).sum(axis=-1)).min(axis=-1),
                np.sqrt((temp_distance[1, :, 0, :] ** 2).sum(axis=-1)).min(axis=-1),
            ]).mean()
            distance2_atoms_order = np.concatenate([
                np.sqrt((temp_distance[0, :, 1, :] ** 2).sum(axis=-1)).argmin(axis=-1),
                np.sqrt((temp_distance[1, :, 0, :] ** 2).sum(axis=-1)).argmin(axis=-1),
            ])

            distance = distance1 if distance1 < distance2 else distance2
            temp_atoms_order = distance1_atoms_order if distance1 < distance2 else distance2_atoms_order
            if distance < min_distance:
                min_distance = distance
                best_symop = symop
                bool_trans = 0 if distance1 < distance2 else 1
                atoms_order = temp_atoms_order

    if min_distance >= 1e-8:
        return None, None

    return np.array([bool_trans, best_symop[0], best_symop[1], best_symop[2]]), atoms_order


class InputMaker:
    """Generate monomer/dimer inputs and ``structure.json`` from a parsed CIF.

    Parameters
    ----------
    reader : CifReader
        Parsed CIF data.
    cpu : int
        Number of CPU cores to request in Gaussian inputs.
    mem : int
        Memory in GB to request in Gaussian inputs.
    method : str
        DFT method/basis string (e.g. ``"PBEPBE/6-31G(d,p)"``).
    engine : str, optional
        Calculation engine. ``"g16"``/``"g09"`` produce ``.gjf`` inputs; any
        other value (``"pyscf"``, ``"gpu4pyscf"``, ``"orca"``) produces ``.xyz``
        geometry files, by default ``"g16"``.
    expand_range : int, optional
        Supercell expansion range for the neighbor search; produces a
        ``(2 * expand_range + 1) ** 3`` supercell, by default 2 (i.e. 5x5x5).
    vdw_margin : float, optional
        Extra distance (Angstrom) added to the sum of VdW radii in the contact
        criterion, by default 0.7.
    """

    def __init__(
        self,
        reader: CifReader,
        cpu: int,
        mem: int,
        method: str,
        engine: str = "g16",
        expand_range: int = 2,
        vdw_margin: float = 0.7,
    ) -> None:
        self.reader = reader
        self.cpu = cpu
        self.mem = mem
        self.method = method
        self.engine = engine
        self.expand_range = expand_range
        self.vdw_margin = vdw_margin
        self.name = reader.basename
        self._is_gaussian = engine in GAUSSIAN_ENGINES
        self._ext = "gjf" if self._is_gaussian else "xyz"

    def generate(self, save_dir: Union[str, Path]) -> dict:
        """Generate all monomer/dimer inputs and write ``structure.json``.

        Parameters
        ----------
        save_dir : str or pathlib.Path
            Directory of the structure. ``inputs/`` is created beneath it and
            ``structure.json`` is written directly into it.

        Returns
        -------
        dict
            The structure metadata that was serialized to ``structure.json``.
        """
        save_dir = Path(save_dir)
        inputs_dir = save_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        reader = self.reader
        n_atoms = len(reader.unique_symbols[0])

        weights = {
            idx: np.array([ATOMIC_WEIGHTS[s] for s in syms])
            for idx, syms in reader.unique_symbols.items()
        }

        # ----------------------------- monomers ----------------------------- #
        sites: List[dict] = []
        for idx, syms in reader.unique_symbols.items():
            cart = reader.convert_frac_to_cart(reader.unique_coords[idx])
            file_name = f"{self.name}_monomer_{idx:03d}"
            self._make_input(inputs_dir, file_name, syms, cart, f"{self.name} monomer {idx}")
            charge, spin = self._charge_spin(syms)
            sites.append({
                "idx": int(idx),
                "symbols": [str(s) for s in syms],
                "frac": np.asarray(reader.unique_coords[idx]).tolist(),
                "charge": charge,
                "spin": spin,
            })

        # ------------------------------ dimers ------------------------------ #
        expanded = reader.expand_mols(expand_range=self.expand_range)

        pairs: List[dict] = []
        dimer_types: List[dict] = []
        moments = np.empty((0, 3))
        distances = np.empty((0,))
        dimers_coo = np.empty((0, 2, n_atoms, 3))
        dimer_type_count = 0

        for cent_idx in reader.unique_symbols:
            cent_cart = reader.convert_frac_to_cart(reader.unique_coords[cent_idx])
            cent_syms = reader.unique_symbols[cent_idx]
            cent_w = weights[cent_idx]

            for ijk, mol_dict in expanded.items():
                for nei_idx, mol_data in mol_dict.items():
                    # Skip self-pair (same molecule, origin unit cell).
                    if ijk == (0, 0, 0) and nei_idx == cent_idx:
                        continue

                    nei_syms = mol_data[0]
                    nei_cart = reader.convert_frac_to_cart(mol_data[1])
                    nei_w = weights[nei_idx]

                    if not check_vdw_contact(cent_cart, cent_syms, nei_cart, nei_syms, self.vdw_margin):
                        continue

                    # Dimer: molecule A atoms followed by molecule B atoms.
                    dimer_coo_flat = np.concatenate([cent_cart, nei_cart], axis=0)
                    dimer_w = np.concatenate([cent_w, nei_w])

                    moment, _ = cal_I(dimer_coo_flat, dimer_w)
                    moment = np.sort(moment).reshape(1, 3)

                    cent_center = cal_center_of_weight(cent_cart, cent_w)
                    nei_center = cal_center_of_weight(nei_cart, nei_w)
                    distance = float(np.linalg.norm(cent_center - nei_center))

                    if moments.shape[0] > 0:
                        mom_cond = (np.abs(moments - moment) <= 1e-2).all(axis=1)
                        dis_cond = np.abs(distances - distance) <= 1e-7
                        match = np.where(mom_cond & dis_cond)[0]
                    else:
                        match = np.array([], dtype=np.int64)

                    ti_flg = None
                    atoms_order_list: Optional[List[int]] = None
                    for cand_idx in match:
                        bool_trans, atoms_order = compare_coordinates(
                            np.array([cent_cart, nei_cart]), dimers_coo[cand_idx]
                        )
                        if bool_trans is None:
                            continue
                        ti_flg = int(cand_idx)
                        if atoms_order is not None:
                            need_sort = 1
                            atoms_order_list = atoms_order.astype(np.int64).tolist()
                        else:
                            need_sort = 0
                        break

                    if ti_flg is None:
                        if len(match) > 0:
                            logger.warning(
                                f"No identical dimer was found for pair "
                                f"(cent={cent_idx}, ijk={ijk}, nei={nei_idx}); "
                                "creating a new dimer type."
                            )
                        ti_flg = dimer_type_count
                        dimer_syms = np.concatenate([cent_syms, nei_syms])
                        file_name = f"{self.name}_dimer_{ti_flg:03d}"
                        comment = f"{self.name} dimer {ti_flg} (sites {cent_idx}-{nei_idx})"
                        self._make_input(inputs_dir, file_name, dimer_syms, dimer_coo_flat, comment)

                        moments = np.concatenate([moments, moment], axis=0)
                        distances = np.concatenate([distances, np.array([distance])])
                        dimers_coo = np.concatenate(
                            [dimers_coo, np.array([cent_cart, nei_cart]).reshape(1, 2, n_atoms, 3)]
                        )
                        bool_trans = np.array([0, 1, 1, 1], dtype=np.int64)
                        need_sort = 0
                        dimer_types.append({
                            "ti_flg": int(ti_flg),
                            "sites": [int(cent_idx), int(nei_idx)],
                            "distance": distance,
                            "inertia": moment.reshape(-1).tolist(),
                        })
                        dimer_type_count += 1

                    pair = {
                        "cent": int(cent_idx),
                        "nei": int(nei_idx),
                        "ijk": [int(ijk[0]), int(ijk[1]), int(ijk[2])],
                        "ti_flg": int(ti_flg),
                        "transpose": int(bool_trans[0]),
                        "signs": [int(bool_trans[1]), int(bool_trans[2]), int(bool_trans[3])],
                        "need_sort": int(need_sort),
                    }
                    if atoms_order_list is not None:
                        pair["atoms_order"] = atoms_order_list
                    pairs.append(pair)

        structure = {
            "name": self.name,
            "method": self.method,
            "engine": self.engine,
            "generation": {"expand_range": self.expand_range, "vdw_margin": self.vdw_margin},
            "lattice": np.asarray(reader.lattice).tolist(),
            "sites": sites,
            "dimer_types": dimer_types,
            "pairs": pairs,
        }
        with open(save_dir / "structure.json", "w") as f:
            json.dump(structure, f, indent=2)

        print(
            f"Generated {len(sites)} monomers and {dimer_type_count} dimer types "
            f"({len(pairs)} pairs) in {inputs_dir}"
        )
        return structure

    def _charge_spin(self, symbols: List[str]) -> Tuple[int, int]:
        """Determine charge and spin multiplicity for a closed/open shell molecule.

        Parameters
        ----------
        symbols : list of str
            Element symbols of the molecule.

        Returns
        -------
        tuple of (int, int)
            ``(charge, spin)``. Charge is always 0; spin is 1 for an even
            electron count and 2 otherwise (matching ``GjfMaker``).
        """
        electrons = sum(GjfMaker.ELEMENTS_NUM[s] for s in symbols)
        spin = 1 if electrons % 2 == 0 else 2
        return 0, spin

    def _make_input(
        self,
        inputs_dir: Path,
        file_name: str,
        symbols: np.ndarray,
        cart_coords: np.ndarray,
        comment: str,
    ) -> None:
        """Write a single input file, choosing the format from the engine.

        Parameters
        ----------
        inputs_dir : pathlib.Path
            Directory where the input file is written.
        file_name : str
            Base name (without extension).
        symbols : np.ndarray
            Element symbols for all atoms.
        cart_coords : np.ndarray, shape (n_atoms, 3)
            Cartesian coordinates in Angstroms.
        comment : str
            Comment line used by the ``.xyz`` writer (ignored for ``.gjf``).
        """
        if self._is_gaussian:
            self._make_gjf(inputs_dir, file_name, symbols, cart_coords)
        else:
            self._make_xyz(inputs_dir, file_name, symbols, cart_coords, comment)

    def _make_gjf(
        self,
        inputs_dir: Path,
        file_name: str,
        symbols: np.ndarray,
        cart_coords: np.ndarray,
    ) -> None:
        """Write a two-link Gaussian input file.

        The first link runs a single-point calculation; the second runs the
        population analysis (``Pop=Full``, ``IOP(3/33=4,5/33=3)``) used to
        extract MO/Fock/overlap matrices.

        Parameters
        ----------
        inputs_dir : pathlib.Path
            Directory where the ``.gjf`` file is written.
        file_name : str
            Base name (without extension).
        symbols : np.ndarray
            Element symbols for all atoms.
        cart_coords : np.ndarray, shape (n_atoms, 3)
            Cartesian coordinates in Angstroms.
        """
        gjf = GjfMaker()
        gjf.set_resource(self.cpu, self.mem)
        gjf.set_function(self.method)
        gjf.add_root("Symmetry=None")
        gjf.set_symbols([str(s) for s in symbols])
        gjf.set_coordinates(np.asarray(cart_coords).tolist())
        gjf.set_title(self.name)
        gjf.create_chk_file()
        gjf.add_link()
        # Link 1: population analysis for matrix extraction.
        gjf.set_function(self.method)
        gjf.add_root("Symmetry=None")
        gjf.add_root("Pop=Full")
        gjf.add_root("IOP(3/33=4,5/33=3)")
        gjf.export_gjf(file_name, save_dir=str(inputs_dir), chk_rwf_name=file_name)

    def _make_xyz(
        self,
        inputs_dir: Path,
        file_name: str,
        symbols: np.ndarray,
        cart_coords: np.ndarray,
        comment: str,
    ) -> None:
        """Write a plain ``.xyz`` geometry file.

        Parameters
        ----------
        inputs_dir : pathlib.Path
            Directory where the ``.xyz`` file is written.
        file_name : str
            Base name (without extension).
        symbols : np.ndarray
            Element symbols for all atoms.
        cart_coords : np.ndarray, shape (n_atoms, 3)
            Cartesian coordinates in Angstroms.
        comment : str
            Comment line written on the second line of the ``.xyz`` file.
        """
        xyz = FileIO()
        xyz.add_symbols([str(s) for s in symbols])
        xyz.add_coordinates(np.asarray(cart_coords))
        xyz.export_xyz_file(str(inputs_dir / f"{file_name}.xyz"), comment=comment)
