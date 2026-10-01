"""Execute every notebook (``pixi run test-notebooks``)."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_executes(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook, timeout=900, resources={"metadata": {"path": str(path.parent)}}
    ).execute()
