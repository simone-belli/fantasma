"""Every notebook in notebooks/ runs top to bottom without errors."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("matplotlib")
pytest.importorskip("ipykernel")

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))

pytestmark = pytest.mark.notebooks


def test_one_notebook_per_model():
    assert len(NOTEBOOKS) == 7


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(nb, timeout=120, kernel_name="python3").execute(cwd=str(path.parent))
