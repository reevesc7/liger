import pytest
from pytest import Config, Item
import os
import shutil
import subprocess
from pathlib import Path


# pyright: reportUnusedParameter=false


SLURM_CMDS = [
    "sacct",
    "sbatch",
    "scancel",
    "squeue",
    "srun",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = "tests"
EXAMPLES_DIR = "examples"
EXAMPLE_DATA_NAME = "example_data.csv"


def _slurm_available() -> bool:
    """True if Slurm CLI tools are present on PATH."""
    if any(shutil.which(cmd) is None for cmd in SLURM_CMDS):
        return False
    vslurm_present = os.environ.get("VSLURM_PRESENT") == "1"
    vslurm_active = subprocess.run(
        ["pgrep", "slurmctld"],
        capture_output=True,
    ).returncode == 0
    if vslurm_present and not vslurm_active:
        return False
    return True


def _handle_slurm_skip(items: list[Item]) -> None:
    if _slurm_available():
        return
    skip_slurm = pytest.mark.skip(reason="Slurm tools not found/enabled")
    for item in items:
        if "slurm" in item.keywords:
            item.add_marker(skip_slurm)


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    _handle_slurm_skip(items)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return PROJECT_ROOT / TESTS_DIR


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return PROJECT_ROOT / EXAMPLES_DIR


@pytest.fixture(scope="session")
def example_data_file() -> Path:
    return PROJECT_ROOT / EXAMPLES_DIR / EXAMPLE_DATA_NAME
