from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

project = "NeuroVLM"
author = "NeuroVLM Team"
copyright = f"{datetime.datetime.now().year}, {author}"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
include_patterns = [
    "index.md",
    "installation.md",
    "api.rst",
    "tutorials/index.md",
    "tutorials/01_intro.ipynb",
    "generated/**",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Generate stubs only from the main API page to avoid recursive autosummary
# processing of generated method-level stubs.
autosummary_generate = ["api.rst"]
autodoc_typehints = "description"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = "off"
nb_execution_timeout = 120
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Only mock optional deps that may be absent in docs builds.
autodoc_mock_imports = [
    "ollama",
]
autosummary_mock_imports = autodoc_mock_imports

html_theme = "sphinx_book_theme"
html_title = "NeuroVLM"
html_static_path = ["_static", "img"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "image_light": "_static/logo_white_bg.png",
        "image_dark": "_static/logo_black_bg.png",
        "text": "",
        "alt_text": "NeuroVLM",
    },
    "repository_url": "https://github.com/neurovlm/neurovlm",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 2,
}
