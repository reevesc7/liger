import pytest
import os
from enum import IntEnum, auto
import subprocess
import time
from pathlib import Path
from liger.run import tpot
from liger.tpot import TPOTManager


PROJ_ROOT = Path(__file__).parents[3]
EG_DIR = PROJ_ROOT / "examples/tpot"
EG_CONFIG_NAME = "example_config_short.json"
EG_SLURM_PROFILE_NAME = "example_slurm_profile.sh"
POPULATION_NAME = "population.pkl"
RUN_OUTPUT_NAMES = ["individuals.csv", "instances.json"]
SLURM_RUN_STATES = {"PENDING", "RUNNING"}
SLURM_FAIL_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}


os.chdir(EG_DIR)


class SlurmChainState(IntEnum):
    RUNNING = auto()
    COMPLETE = auto()
    FAILED = auto()


def test_local(tmp_path: Path) -> None:
    output_dir = tpot.init_tpot_dir(EG_CONFIG_NAME, output_dir=tmp_path)
    assert (output_dir / tpot.CONFIG_NAME).is_file()
    assert (output_dir / tpot.PARAMS_NAME).is_file()
    complete = tpot.run_local_segment(output_dir)
    assert (output_dir / TPOTManager.INDIVS_NAME).is_file()
    assert (output_dir / TPOTManager.INSTANCES_NAME).is_file()
    while not complete:
        assert (output_dir / TPOTManager.POP_NAME).is_file()
        complete = tpot.run_local_segment(output_dir)
        assert (output_dir / TPOTManager.INDIVS_NAME).is_file()
        assert (output_dir / TPOTManager.INSTANCES_NAME).is_file()


def _slurm_chain_state(dir: Path) -> SlurmChainState:
    job_ids = [
        path.stem.rsplit("-", 1)[1]
        for path in dir.glob(str(tpot.SLURM_OUT_NAME).replace("%j", "*"))
    ]
    sacct_result = subprocess.run([
        "sacct",
        "-j",
        ",".join(job_ids),
        "--format=JobID,State",
        "--noheader",
        "--parsable2",
    ], capture_output=True, text=True, check=True)
    states = [
        line.split("|", 1)[1]
        for line in sacct_result.stdout.strip().splitlines()
        if "." not in line.split("|", 1)[0]
    ]
    if any(state in SLURM_FAIL_STATES for state in states):
        return SlurmChainState.FAILED
    if any(state in SLURM_RUN_STATES for state in states):
        return SlurmChainState.RUNNING
    return SlurmChainState.COMPLETE


@pytest.mark.slurm
def test_slurm(tmp_path: Path) -> None:
    output_dir = tpot.init_tpot_dir(
        EG_CONFIG_NAME,
        slurm_profile_path=EG_SLURM_PROFILE_NAME,
        output_dir=tmp_path,
    )
    assert (output_dir / tpot.CONFIG_NAME).is_file()
    assert (output_dir / tpot.PARAMS_NAME).is_file()
    assert (output_dir / tpot.SLURM_PROFILE_NAME).is_file()
    job_id = tpot.run_slurm_segment(output_dir, recurse=True)
    assert job_id is not None
    while True:
        time.sleep(4)
        state = _slurm_chain_state(output_dir)
        if state is SlurmChainState.COMPLETE:
            break
        assert state is not SlurmChainState.FAILED
    assert (output_dir / TPOTManager.INDIVS_NAME).is_file()
    assert (output_dir / TPOTManager.INSTANCES_NAME).is_file()
