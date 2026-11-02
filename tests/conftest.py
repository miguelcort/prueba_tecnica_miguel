from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.pipeline import main

ROOT = Path(__file__).resolve().parent.parent


def make_project(base: Path) -> Path:
    """A clean project with only configs and originals: what a new machine would have."""
    shutil.copytree(ROOT / "configs", base / "configs")
    shutil.copytree(ROOT / "docs_raw", base / "docs_raw")
    return base


@pytest.fixture(scope="session")
def built_project(tmp_path_factory) -> Path:
    project = make_project(tmp_path_factory.mktemp("first"))
    assert main(["--config", str(project / "configs" / "pipeline.yaml")]) == 0
    return project
