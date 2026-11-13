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
from liger.context import Context
from ._manager import TPOTManager
from . import data, paths


def init_tpot_dir(
    config_path: str | Path,
    *,
    slurm_profile_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    context = Context()
    tpot: TPOTManager = data.load_build(config_path)(context)
    if output_dir is not None:
        tpot.set_output_dir(output_dir)
    tpot.output_dir_.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, paths.config_path(tpot.output_dir_))
    context.freeze(paths.context_path(tpot.output_dir_))
    if slurm_profile_path is not None:
        slurm_profile_path = Path(slurm_profile_path)
        shutil.copy(slurm_profile_path, paths.slurm_profile_path(tpot.output_dir_))
    print(f"Initialized TPOT run directory:\n{tpot.output_dir_}\n")
    return tpot.output_dir_


def run_local_segment(checkpoint_dir: str | Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    context = Context.from_frozen(paths.context_path(checkpoint_dir))
    tpot: TPOTManager = data.load_build(paths.config_path(checkpoint_dir))(context)
    context.check_used()
    if not tpot.output_dir_.is_dir() or not tpot.output_dir_.samefile(checkpoint_dir):
        tpot.set_output_dir(checkpoint_dir)
    tpot.run_segment()
    return tpot.is_complete()


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
