# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import shutil
from pathlib import Path
import subprocess
import json
from datetime import datetime, timezone
from liger.typing import LgConfig
import liger.config as cfg
from liger.tpot import TPOTManager


DATETIME_FMT = "%Y-%m-%d_%H-%M-%S.%f"
SLURM_PROFILE_NAME = Path("slurm_profile.sh")
CONFIG_NAME = Path("config.json")
PARAMS_NAME = Path("params.json")


def init_tpot_dir(
    config_path: str | Path,
    slurm_profile_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    config: dict[str, LgConfig] = json.load(config_path.open())
    now = datetime.now(timezone.utc).strftime(DATETIME_FMT)
    if output_dir is not None:
        output_dir = Path(output_dir)
        config["output_dir"] = str(output_dir / now)
    elif "output_dir" in config:
        config["output_dir"] = str(Path(str(config["output_dir"])) / now)
    else:
        config["output_dir"] = now
    tpot = cfg.parse_config(config, TPOTManager)
    tpot.output_dir_.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, tpot.output_dir_ / CONFIG_NAME)
    json.dump(cfg.to_config(tpot), (tpot.output_dir_ / PARAMS_NAME).open("w"), indent=4)
    if slurm_profile_path is not None:
        slurm_profile_path = Path(slurm_profile_path)
        shutil.copy(slurm_profile_path, tpot.output_dir_ / SLURM_PROFILE_NAME)
    print(f"Initialized TPOT run directory:\n{tpot.output_dir_}\n")
    return tpot.output_dir_


def run_local_segment(checkpoint_dir: str | Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    params_path = checkpoint_dir / PARAMS_NAME
    config: dict[str, LgConfig] = json.load((params_path).open())
    if Path(str(config["output_dir"])).resolve() != checkpoint_dir:
        config["output_dir"] = str(checkpoint_dir)
        json.dump(config, params_path.open("w"), indent=4)
    tpot = cfg.parse_config(config, TPOTManager)
    tpot.run_segment()
    return True if tpot.is_complete_ else False


def _parse_slurm_options(filepath: Path) -> list[str]:
    if not filepath.is_file():
        return []
    return [
        line.split("#SBATCH ", 1)[-1].strip()
        for line in filepath.read_text().split("\n")
        if line.strip().startswith("#SBATCH ")
    ]


def _submit_slurm_segment(
    checkpoint_dir: Path,
    recurse: bool,
    job_id: str | None = None,
) -> int:
    slurm_profile_path = checkpoint_dir / SLURM_PROFILE_NAME
    slurm_options = _parse_slurm_options(slurm_profile_path) + [
        f"--job-name=lgtpot_{checkpoint_dir.name}",
        f"--output={checkpoint_dir}/slurm-%j.out",
    ]
    if job_id:
        slurm_options.append(f"--dependency=afterany:{job_id}")
    return int(subprocess.run(
        ["sbatch"] + slurm_options,
        input=f"#!/bin/bash --login\n"
            f"source {slurm_profile_path}\n"
            f"liger tpot run slurm {checkpoint_dir}{' --recurse' if recurse else ''}\n",
        capture_output=True,
        text=True,
    ).stdout.strip().rsplit(" ", 1)[-1])


def run_slurm_segment(checkpoint_dir: str | Path, recurse: bool) -> int | None:
    checkpoint_dir = Path(checkpoint_dir)
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return _submit_slurm_segment(checkpoint_dir, recurse)
    if not recurse:
        run_local_segment(checkpoint_dir)
        return None
    next_job_id = _submit_slurm_segment(checkpoint_dir, recurse, job_id)
    try:
        tpot_is_complete = run_local_segment(checkpoint_dir)
    except BaseException:
        subprocess.run(["scancel", str(next_job_id)])
        raise
    if tpot_is_complete:
        subprocess.run(["scancel", str(next_job_id)])
    return next_job_id
