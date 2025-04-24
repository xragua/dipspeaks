import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root to sys.path so Sphinx can import your package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -- Project information -----------------------------------------------------
project = 'dipspeaks'
author = 'Graciela Sanjurjo Ferrin'
# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# Sphinx extensions for autodoc, summaries, notebook integration, etc.
extensions = [
    'sphinx.ext.autodoc',       # Core: pull in docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.autosummary',   # Auto-generate API docs
    #'sphinx.ext.napoleon',      # Support NumPy/Google style docstrings
    'nbsphinx',                 # Render Jupyter notebooks
]

# Automatically generate stub files for autosummary
autosummary_generate = True

# Paths to templates and static assets
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# Use the Read the Docs theme (ensure it's installed in your environment)
html_theme = 'sphinx_rtd_theme'

# If needed, specify where to find the theme
try:
    import sphinx_rtd_theme
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    html_theme_path = []

# Logo and static files
html_logo = '_static/logo.png'
html_static_path = ['_static']

# Custom CSS (if you have any overrides)
# html_css_files = ['custom.css']
