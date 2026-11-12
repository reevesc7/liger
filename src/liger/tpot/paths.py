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


from pathlib import Path


SLURM_PROFILE = "slurm_profile.sh"
CONFIG = "config.json"
PARAMS = "params.json"
SLURM_OUT = "slurm-%j.out"
POP = "population.pkl"
INDIVS = "individuals.csv"
INSTANCES = "instances.json"


def path(checkpoint_dir: str | Path, path: str | Path) -> Path:
    return Path(checkpoint_dir) / path


def slurm_profile_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, SLURM_PROFILE)


def config_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, CONFIG)


def params_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, PARAMS)


def slurm_out_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, SLURM_OUT)


def pop_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, POP)


def indivs_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, INDIVS)


def instances_path(checkpoint_dir: str | Path) -> Path:
    return path(checkpoint_dir, INSTANCES)
