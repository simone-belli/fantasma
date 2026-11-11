"""Guard: the version is consistent across the project.

``pyproject.toml`` is the single source of truth. ``fantasma.__version__`` is read from the
installed package metadata, so these tests fail if the metadata is stale (the version was bumped
without re-running ``poetry install``) or if a hardcoded version string creeps back in.
"""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import fantasma

ROOT = Path(__file__).parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
HINT = "pyproject.toml is the source of truth; re-run `poetry install` after changing it."


def test_pyproject_version_is_pep440():
    # Release versions, optionally with a pre-, post- or dev-release suffix.
    pattern = r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?"
    assert re.fullmatch(pattern, PYPROJECT), f"not a PEP 440 version: {PYPROJECT!r}"


def test_package_version_matches_pyproject():
    assert fantasma.__version__ == PYPROJECT, HINT


def test_installed_metadata_matches_pyproject():
    assert version("fantasma") == PYPROJECT, HINT


def test_no_hardcoded_version_in_sources():
    # Any version literal in the package would drift from pyproject.toml on the next bump.
    literal = re.compile(r"""__version__\s*=\s*["']\d+\.\d+""")
    offenders = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        if literal.search(p.read_text())
    ]
    assert not offenders, f"hardcoded __version__ in {offenders}; {HINT}"
