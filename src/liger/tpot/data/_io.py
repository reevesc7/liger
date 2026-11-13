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


import hashlib
import importlib.util
import inspect
from pathlib import Path
import json
import sys
from typing import Callable
import dill
import pandas as pd
from tpot import Population
from liger import config as cfg
from liger.tpot import paths
from ._schema import IndivCol
from . import _parse


def load_build(config_path: Path) -> Callable:
    name = "_liger_tpot_config_" + hashlib.sha1(
        str(config_path).encode()
    ).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not parse proper spec from {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if not (callable(build) and len(inspect.signature(build).parameters.keys()) == 1):
        raise TypeError(f"{config_path}: expected a callable 'build(context)'")
    return build


def load_pop(checkpoint_dir: str | Path) -> Population:
    pop_path = paths.pop_path(checkpoint_dir)
    pop = dill.load(pop_path.open("rb"))
    if not isinstance(pop, Population):
        raise TypeError(f"{pop_path} does not contain a "
            f"{Population.__name__!r} object")
    return _parse.parse_pop(pop)


def dump_pop(pop: Population, checkpoint_dir: str | Path) -> None:
    dill.dump(_parse.parse_pop(pop), paths.pop_path(checkpoint_dir).open("wb"))


def load_indivs(checkpoint_dir: str | Path) -> pd.DataFrame:
    indivs = pd.read_csv(paths.indivs_path(checkpoint_dir), index_col=IndivCol.ID)
    indivs[IndivCol.INSTANCE] = cfg.parse_config(
        json.load(paths.instances_path(checkpoint_dir).open()),
        pd.Series,
    )
    return _parse.parse_indivs(indivs)


def dump_indivs(indivs: pd.DataFrame, checkpoint_dir: str | Path) -> None:
    _parse.parse_indivs(indivs).drop(
        IndivCol.INDIV,
        axis=1,
        errors="ignore",
    ).to_csv(paths.indivs_path(checkpoint_dir))
    instances = cfg.to_config(indivs[IndivCol.INSTANCE])
    json.dump(instances, paths.instances_path(checkpoint_dir).open("w"), indent=4)
