# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the source tree so autodoc can import the package
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Project information ------------------------------------------------------
project = "brain-fwi"
author = "Morgan Hough"
copyright = "2024-2026, Morgan Hough"
release = "0.1.0"
version = "0.1"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# MyST extensions for rich Markdown (math, admonitions, etc.)
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstrings = False
napoleon_numpy_docstrings = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Mock imports for packages that require compiled backends, GPU, or special licenses.
# numpy and scipy are NOT mocked because module-level code in properties.py
# uses np.zeros with item assignment, which breaks with mock objects.
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "jwave",
    "jaxdf",
    "equinox",
    "optax",
    "matplotlib",
    "nibabel",
    "h5py",
    "brainweb_dl",
    "tqdm",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output --------------------------------------------------
html_theme = "furo"
html_title = "brain-fwi"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/mhough/brain-fwi",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Source suffix / parsers ---------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Suppress warnings for missing references to mocked modules
suppress_warnings = ["ref.myst"]
