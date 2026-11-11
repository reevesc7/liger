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


from . import _config, data, objectives, paths
from ._manager import TPOTManager
from ._params import (
    Objective,
    InverseObjectives,
    DatasetParams,
    EvolutionParams,
    EvalParams,
    EndParams,
    RuntimeParams,
)
from ._run import init_tpot_dir, run_local_segment, run_slurm_segment


__all__ = [
    "data",
    "objectives",
    "paths",
    "TPOTManager",
    "Objective",
    "InverseObjectives",
    "DatasetParams",
    "EvolutionParams",
    "EvalParams",
    "EndParams",
    "RuntimeParams",
    "init_tpot_dir",
    "run_local_segment",
    "run_slurm_segment",
]


_config.register()
del _config
