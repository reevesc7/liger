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


# /// --- Fixes missing pkg_resources, imported by stopit --- \\\
import importlib.util
if importlib.util.find_spec("pkg_resources") is None:
    from types import ModuleType
    from importlib.metadata import version
    import sys
    fake_pkg_resources = ModuleType("pkg_resources")
    setattr(
        fake_pkg_resources,
        "get_distribution",
        lambda name: type("Distribution", (), {"version": version(name)})(),
    )
    sys.modules["pkg_resources"] = fake_pkg_resources
# \\\ --------------------------------------------------------///


from ._params import (
    Objective,
    InverseObjectives,
    DatasetParams,
    EvolutionParams,
    EvalParams,
    EndParams,
    RuntimeParams,
)
from ._manager import TPOTManager
from . import _config


__all__ = [
    "Objective",
    "InverseObjectives",
    "DatasetParams",
    "EvolutionParams",
    "EvalParams",
    "EndParams",
    "RuntimeParams",
    "TPOTManager",
]


_config.register()
del _config
