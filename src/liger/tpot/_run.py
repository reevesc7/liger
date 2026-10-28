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
import liger.config as cfg
from liger.config.typing import Config
from ._manager import TPOTManager
from . import data, paths


def init_tpot_dir(
    config_path: str | Path,
    *,
    slurm_profile_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    config: dict[str, Config] = json.load(config_path.open())
    dirname = "liger_tpot_" + datetime.now(timezone.utc).strftime(paths.DATETIME_FMT)
    if output_dir is not None:
        output_dir = Path(output_dir)
        config["output_dir"] = str(output_dir / dirname)
    elif "output_dir" in config:
        config["output_dir"] = str(Path(str(config["output_dir"])) / dirname)
    else:
        config["output_dir"] = dirname
    tpot = cfg.parse_config(config, TPOTManager)
    tpot.output_dir_.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, paths.config_path(tpot.output_dir_))
    data.dump_params(cfg.to_config(tpot), tpot.output_dir_)
    if slurm_profile_path is not None:
        slurm_profile_path = Path(slurm_profile_path)
        shutil.copy(slurm_profile_path, paths.slurm_profile_path(tpot.output_dir_))
    print(f"Initialized TPOT run directory:\n{tpot.output_dir_}\n")
    return tpot.output_dir_


def run_local_segment(checkpoint_dir: str | Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config = data.load_params(checkpoint_dir)
    tpot = cfg.parse_config(config, TPOTManager)
    if tpot.output_dir_ != checkpoint_dir:
        tpot.output_dir_ = checkpoint_dir
        data.dump_params(cfg.to_config(tpot), checkpoint_dir)
    tpot.run_segment()
    return True if tpot.is_complete_ else False


def _submit_slurm_segment(
    checkpoint_dir: Path,
    recurse: bool,
    job_id: str | None = None,
) -> int:
    slurm_profile_path = paths.slurm_profile_path(checkpoint_dir)
    if slurm_profile_path.is_file():
        slurm_options = data.parse_slurm_options(slurm_profile_path.read_text())
    else:
        slurm_options = []
    slurm_options.extend([
        f"--job-name={checkpoint_dir.name}",
        f"--output={paths.slurm_out_path(checkpoint_dir)}",
        "--parsable",
        f"--wrap=#!/usr/bin/env bash\n"
            f"source {slurm_profile_path}\n"
            f"liger tpot run slurm {checkpoint_dir}{' --recurse' if recurse else ''}\n",
    ])
    if job_id:
        slurm_options.append(f"--dependency=afterany:{job_id}")
    re = subprocess.run(
        ["sbatch"] + slurm_options,
        capture_output=True,
        text=True,
        check=True,
    )
    return(int(re.stdout))


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
        subprocess.run(["scancel", str(next_job_id)], check=True)
        raise
    if tpot_is_complete:
        subprocess.run(["scancel", str(next_job_id)], check=True)
    return next_job_id
