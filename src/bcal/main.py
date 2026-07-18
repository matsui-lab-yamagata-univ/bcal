import argparse
from pathlib import Path
from time import time

import numpy as np
from mcal import CifReader
from tcal import Tcal

from bcal.calculations.bcal import Bcal, HIGH_SYMMETRY_POINTS
from bcal.utils.input_maker import InputMaker
from bcal.utils.log import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inputs from a CIF, run DFT, and compute band structure / effective mass."
    )
    parser.add_argument("file", help="cif file path", type=str)
    parser.add_argument(
        "-M", "--method",
        help="calculation method (default: PBEPBE/6-31G(d,p))",
        type=str,
        default="PBEPBE/6-31G(d,p)",
    )
    parser.add_argument(
        "-c", "--cpu",
        help="setting the number of cpu (default is 4)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-m", "--mem",
        help="setting the number of memory [GB] (default is 10 GB)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-o", "--output",
        help="output directory (default is the directory of the cif file)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-r", "--read",
        help="read existing log files without running DFT calculations",
        action="store_true",
    )
    parser.add_argument(
        "--engine",
        help="calculation engine (default is g16)",
        type=str,
        default="g16",
        choices=["g16", "g09", "pyscf", "gpu4pyscf", "orca"],
    )
    parser.add_argument(
        "--resume",
        help="resume the calculation from the last step",
        action="store_true",
    )
    parser.add_argument(
        "--num-mo",
        help=f"number of MOs per molecule on each side of the frontier (HOMO-side and LUMO-side); total MOs = 2 * num_mo (default is 15)",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--band-path",
        help=(
            "high-symmetry k-path for the band diagram, given as a sequence of "
            "single-character labels (G = Gamma). "
            f"available: {' '.join(sorted(HIGH_SYMMETRY_POINTS))} (default: XGYGZ)"
        ),
        type=str,
        default="XGYGZ",
    )
    parser.add_argument(
        "--bse",
        help="use Basis Set Exchange (PySCF/gpu4pyscf only)",
        action="store_true",
    )
    args = parser.parse_args()

    configure_logging()

    unknown = sorted(set(args.band_path) - set(HIGH_SYMMETRY_POINTS))
    if unknown:
        parser.error(
            f"Unknown high-symmetry labels in --band-path: {unknown}. "
            f"Available: {' '.join(sorted(HIGH_SYMMETRY_POINTS))}"
        )

    print("----------------------------------------")
    print(" bcal 0.1.0 (2026/06/18) by Matsui Lab. ")
    print("----------------------------------------")
    print(f"\nInput File Name: {args.file}")
    Tcal.print_timestamp()
    start_time = time()
    print()

    reader = CifReader(args.file)
    output_dir = Path(args.output) if args.output is not None else Path(args.file).parent
    save_dir = output_dir / reader.basename

    if args.engine == "pyscf" or args.engine == "gpu4pyscf" or args.engine == "orca":
        if args.method.split("/")[0] == "PBEPBE":
            old_method = args.method
            args.method = args.method.replace("PBEPBE", "PBE", 1)
            logger.warning(
                f"The '{args.engine}' engine does not accept the Gaussian "
                f"functional name 'PBEPBE'; the method was changed from "
                f"'{old_method}' to '{args.method}'."
            )

    # Stage 1: generate inputs/ and structure.json.
    if not args.read:
        InputMaker(
            reader,
            args.cpu,
            args.mem,
            args.method,
            engine=args.engine,
        ).generate(save_dir)

    # Stage 2: DFT -> matrices -> tight-binding -> effective mass.
    bcal = Bcal(
        save_dir,
        num_mo=args.num_mo,
        engine=args.engine,
        cpu=args.cpu,
        mem=args.mem,
        method=args.method,
        bse=args.bse,
    )
    bcal.run_dft(read=args.read, resume=args.resume)
    bcal.cal_effective_mass(band_path=args.band_path)

    print()
    print(f"LUMO band edge: {_format_edge(bcal.lumo_edge_frac, bcal.lumo_edge_label)}")
    print(_format_effective_mass(bcal.lumo_em, bcal.lumo_em_vectors))

    print()
    print(f"HOMO band edge: {_format_edge(bcal.homo_edge_frac, bcal.homo_edge_label)}")
    print(_format_effective_mass(bcal.homo_em, bcal.homo_em_vectors))

    on_path = bcal.band_edges_on_path(args.band_path)
    missing = [name for name, ok in on_path.items() if not ok]
    if missing:
        edges = {
            "HOMO": (bcal.homo_edge_frac, bcal.homo_edge_label),
            "LUMO": (bcal.lumo_edge_frac, bcal.lumo_edge_label),
        }
        detail = ", ".join(f"{name} {_format_edge(*edges[name])}" for name in missing)
        logger.warning(
            f"Band edge(s) not on --band-path '{args.band_path}': {detail}. "
            f"band.png does not show the true extremum for these; "
            f"add their k-points to --band-path."
        )

    print(f"\nSaved results to {save_dir / 'results'}")

    print()
    Tcal.print_timestamp()
    end_time = time()
    elapsed_time = end_time - start_time
    print_elapsed_time(elapsed_time)


def _format_edge(frac: np.ndarray, label: str | None) -> str:
    """Format a band-edge k-point for console output.

    Parameters
    ----------
    frac : numpy.ndarray, shape (3,)
        Fractional reciprocal-lattice coordinates of the band edge.
    label : str or None
        High-symmetry point label, or ``None`` if the k-point is not one.

    Returns
    -------
    str
        ``"<label>  k=(...)"`` when the k-point is a high-symmetry point
        (label and coordinates), otherwise ``"k=(...)"`` (coordinates only).
    """
    coord = f"({frac[0]:+.3f}, {frac[1]:+.3f}, {frac[2]:+.3f})"
    return f"{label}  k={coord}" if label else f"k={coord}"


def _format_effective_mass(masses: np.ndarray, vectors: np.ndarray) -> str:
    """Format principal effective masses and their axis vectors for console output.

    Parameters
    ----------
    masses : numpy.ndarray, shape (3,)
        Principal effective masses (units of electron mass), sorted by ``|m|``
        in ascending order.
    vectors : numpy.ndarray, shape (3, 3)
        Unit principal-axis vectors as rows (Cartesian reciprocal-space),
        ``vectors[i]`` paired with ``masses[i]``.

    Returns
    -------
    str
        A multi-line, indented block listing each mass with its axis vector,
        one line per principal axis.
    """
    lines = []
    for i, (m, v) in enumerate(zip(masses, vectors), start=1):
        vec = f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"
        lines.append(f"  m{i} = {m:+.3f} m_e  v={vec}")
    return "\n".join(lines)


def print_elapsed_time(elapsed_time: float) -> None:
    """Print the elapsed time in a human-readable format."""
    elapsed_time_h = int(elapsed_time // 3600)
    elapsed_time_min = int((elapsed_time % 3600) // 60)
    elapsed_time_sec = int(elapsed_time % 60)
    elapsed_time_ms = int(elapsed_time * 1000)
    if elapsed_time < 1:
        print(f"Elapsed Time: {elapsed_time_ms} ms")
    elif elapsed_time < 60:
        print(f"Elapsed Time: {elapsed_time_sec} sec")
    elif elapsed_time < 3600:
        print(f"Elapsed Time: {elapsed_time_min} min {elapsed_time_sec} sec")
    else:
        print(f"Elapsed Time: {elapsed_time_h} h {elapsed_time_min} min {elapsed_time_sec} sec")


if __name__ == "__main__":
    main()
