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


DATETIME_FMT = "%Y-%m-%d_%H-%M-%S.%f"
SLURM_PROFILE = "slurm_profile.sh"
CONFIG = "config.json"
PARAMS = "params.json"
SLURM_OUT = "slurm-%j.out"
POP = Path("population.pkl")
INDIVS = Path("individuals.csv")
INSTANCES = Path("instances.json")
