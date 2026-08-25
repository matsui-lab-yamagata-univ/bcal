"""Sphinx configuration for the bcal documentation."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the ``bcal`` package importable for autodoc (src layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -----------------------------------------------------
project = "bcal"
author = "Tomoharu Okada, Koki Ozawa, Hiroyuki Matsui"
copyright = "2026, Tomoharu Okada, Koki Ozawa, Hiroyuki Matsui"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []
language = "en"

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# Optional quantum-chemistry backends are imported lazily inside functions;
# mock them so autodoc never fails when a backend is not installed.
autodoc_mock_imports = ["pyscf", "gpu4pyscf", "orca_pi", "basis_set_exchange"]

# -- napoleon (NumPy-style docstrings) ---------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
