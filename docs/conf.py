"""Sphinx configuration."""
from datetime import datetime


project = "Quasi-Anharmonic Analysis"
author = "Timothy H. Click"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "sphinx_rtd_theme",
    "sphinx_automodapi.automodapi",
]
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
numpydoc_show_class_members = False
automodapi_inheritance_diagram = False
