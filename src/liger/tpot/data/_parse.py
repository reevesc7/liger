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


import ast
import pandas as pd
from sklearn.base import BaseEstimator
from tpot import Population
from liger import config as cfg
from liger.config import typing
from ._schema import IndivCol


def parse_slurm_options(slurm_profile: str) -> list[str]:
    return [
        line.split("#SBATCH ", 1)[-1].strip()
        for line in slurm_profile.split("\n")
        if line.strip().startswith("#SBATCH ")
    ]


def _parse_int_pair(value) -> tuple[int, int] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, tuple):
        v1, v2 = value
        return int(v1), int(v2)
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise ValueError(f"Unable to parse {value!r} as '(int, int) | None'")


def _parse_str_seq(value) -> list[str] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return [str(element) for element in value]
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except ValueError:
            return [element.strip() for element in value.split(",")]
    raise ValueError(f"Unable to parse {value!r} as 'list[str] | None'")


def _is_instance_repr(value) -> bool:
    if not isinstance(value, list):
        return False
    for element in value:
        if not isinstance(element, tuple):
            return False
        if len(element) != 2:
            return False
        if not isinstance(element[0], str) or not isinstance(element[1], str):
            return False
    return True


def _parse_instance(value) -> list[tuple[str, str]] | BaseEstimator | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, BaseEstimator):
        return value
    if isinstance(value, str):
        value = ast.literal_eval(value)
        if _is_instance_repr(value):
            return value
    if typing.is_lg_config_like(value):
        return cfg.parse_config(value, BaseEstimator)
    raise ValueError(f"Unable to parse {value!r} as "
        "'list[(str, str)] | BaseEstimator | None'")


def parse_pop(pop: Population) -> Population:
    indivs = pop.evaluated_individuals
    indivs[IndivCol.GEN] = indivs[IndivCol.GEN].astype("Int64")
    return pop


def parse_indivs(indivs: pd.DataFrame) -> pd.DataFrame:
    parsed = indivs.copy().rename_axis(IndivCol.ID)
    parsed[IndivCol.PARENTS] = parsed[IndivCol.PARENTS].map(_parse_int_pair)
    parsed[IndivCol.VAR_FUNC] = parsed[IndivCol.VAR_FUNC].map(_parse_str_seq)
    parsed[IndivCol.GEN] = parsed[IndivCol.GEN].astype("Int64")
    parsed[IndivCol.INSTANCE] = parsed[IndivCol.INSTANCE].map(_parse_instance)
    return parsed
