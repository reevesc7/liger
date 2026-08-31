import pytest
from pytest import Config, Item
import os
import shutil
import subprocess


# pyright: reportUnusedParameter=false


SLURM_CMDS = [
    "sacct",
    "sbatch",
    "scancel",
    "squeue",
    "srun",
]


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers", "slurm: requires Slurm CLI tools within environment scope"
    )


def slurm_available() -> bool:
    """True if Slurm CLI tools are present on PATH."""
    return not any(shutil.which(cmd) is None for cmd in SLURM_CMDS)


def vslurm_active() -> bool:
    pgrep_result = subprocess.run(["pgrep", "slurmctld"], capture_output=True)
    return pgrep_result.returncode == 0


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    if slurm_available():
        if os.environ.get("VSLURM_PRESENT") != "1":
            return
        if vslurm_active():
            return
    # TODO: detect whether Slurm/vslurm daemon is active and condition `reason` on it
    skip_slurm = pytest.mark.skip(reason="Slurm CLI tools not found on PATH")
    for item in items:
        if "slurm" in item.keywords:
            item.add_marker(skip_slurm)
