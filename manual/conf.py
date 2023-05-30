# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "qermit-manual"
author = "Cambridge Quantum Computing Ltd"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "jupyter_sphinx",
]

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/CQCL/qermit",
    "use_repository_button": True,
    "use_issues_button": True,
    "logo": {
        "image_light": "_static/Quantinuum_logo_black.png",
        "image_dark": "_static/Quantinuum_logo_white.png",
    },
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
