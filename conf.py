import os
import sys

# Add source directory to path
sys.path.insert(0, os.path.abspath('../src/openparse'))

project = 'OpenParse'
copyright = '2025'
author = 'Michael Mooring'
release = '0.7.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx.ext.autosummary',
]

# Source file mappings
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Theme settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# AutoDoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Build settings
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Include README.md
root_doc = 'index'