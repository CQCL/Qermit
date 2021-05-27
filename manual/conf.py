# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "qermit-manual"
copyright = "2020-2021 Cambridge Quantum Computing Ltd"
author = "Cambridge Quantum Computing Ltd"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "jupyter_sphinx",
]

html_theme = "sphinx_rtd_theme"
