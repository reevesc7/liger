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
import json
import dill
import pandas as pd
from tpot import Population
from liger import config as cfg
from liger.tpot import paths
from ._schema import Col
from . import _parse


def _pop_path(dir: str | Path) -> Path:
    return Path(dir) / paths.POP


def load_pop(dir: str | Path) -> Population:
    pop_path = _pop_path(dir)
    pop = dill.load(pop_path.open("rb"))
    if not isinstance(pop, Population):
        raise TypeError(f"{pop_path!r} does not contain a "
            f"{Population.__name__!r} object")
    return _parse.parse_pop(pop)


def dump_pop(pop: Population, dir: str | Path) -> None:
    dill.dump(pop, _pop_path(dir).open("wb"))


def _indivs_paths(dir: str | Path) -> tuple[Path, Path]:
    dir = Path(dir)
    return dir / paths.INDIVS, dir / paths.INSTANCES


def load_indivs(dir: str | Path) -> pd.DataFrame:
    indivs_path, instances_path = _indivs_paths(dir)
    indivs = pd.read_csv(indivs_path, index_col=Col.ID)
    indivs[Col.INSTANCE] = cfg.parse_config(json.load(instances_path.open()), pd.Series)
    return _parse.parse_indivs(indivs)


def dump_indivs(indivs: pd.DataFrame, dir: str | Path) -> None:
    indivs_path, instances_path = _indivs_paths(dir)
    indivs.drop(Col.INDIV, axis=1).rename_axis(Col.ID).to_csv(indivs_path)
    instances = cfg.to_config(indivs[Col.INSTANCE])
    json.dump(instances, instances_path.open("w"), indent=4)
