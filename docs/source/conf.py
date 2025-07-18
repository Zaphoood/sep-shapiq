"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from __future__ import annotations

from pathlib import Path
import sys

# Make sure the `shapiq_student` package can be imported. This assumes this
# configuration file lives in a directory "./docs/source" relative to the
# project root.
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "shapiq_student"
copyright = "2025, Milana Gurbanova, Lirona Iseni, Tanja Mursch, Max Neuner, Mathis Weber"
author = "Milana Gurbanova, Lirona Iseni, Tanja Mursch, Max Neuner, Mathis Weber"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",  # Insert links to the source code to classes, methods etc.
    "sphinx.ext.napoleon",  # Enable parsing of Google-style docstrings
    # Note: This requires 'pandoc' to be installed on the system
    "nbsphinx",  # Include Jupyter Notebooks
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["scripts"]

# -- Autodoc ------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,  # Generate documentation for module members
    "special-members": "__init__",  # Also generate docs for __init__ methods
    "inherited-members": False,  # Exclude inherited members from documentation
}
autodoc_member_order = "bysource"
autoclass_content = "class"  # Use the class docstring, not the docstring of its __init__ method

# -- Copybutton

# Exclude prompts (.gp) and outputs (.go) when copying code blocks
copybutton_exclude = ".gp, .go"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "furo"
html_css_files = [
    "custom.css",
]

# -- Jupyter Notebooks -------------------------------------------------------
nbsphinx_prolog = """
.. raw:: html

    <style>
        div.output_area {
            display: flex;
            flex-direction: column;
        }
        div.output_area img {
            align-self: center;
        }
    </style>
"""
