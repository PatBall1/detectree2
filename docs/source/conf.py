# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Add repository root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "detectree2"
copyright = "2023, James Ball"
author = "James Ball"

# The full version, including alpha/beta/rc tags
release = "1.0.8"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",  # section refs
    "sphinx.ext.todo",              # see contributing guide
    "nbsphinx",                     # render notebooks
]

# Autodoc / autosummary settings
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}

# Mock heavy optional dependencies to avoid import failures on docs build
autodoc_mock_imports = [
    "torch",
    "detectron2",
    "detectron2.engine",
    "detectron2.config",
    "detectron2.data",
    "detectron2.layers",
    "detectron2.structures",
    "detectron2.utils",
    "detectron2.evaluation",
    "detectron2.checkpoint",
    "detectron2.model_zoo",
    "cv2",
    "rasterio",
    "geopandas",
    "shapely",
    "fiona",
    "rtree",
    "pycocotools",
]
autosectionlabel_prefix_document = True
todo_include_todos = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# html_css_files = ["css/style.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
