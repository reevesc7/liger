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


from typing import TypeVar
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum
from functools import cached_property
from pathlib import Path
import pandas as pd
from tpot import Population
from liger import config as cfg
from liger import tpot
from . import _io
from ._schema import RunCol, IndivCol, NodeCol, EdgeCol


K = TypeVar("K", bound=Hashable)


class ScoreValuation(StrEnum):
    GREATER = "greater"
    LESSER = "lesser"
    ZERO = "zero"


class GraphsBreakdown:
    def __init__(
        self,
        runs_indivs: Mapping[K, pd.DataFrame],
        score_valuation: ScoreValuation = ScoreValuation.ZERO,
    ) -> None:
        self.score_valuation = score_valuation
        self.indivs = pd.concat(
            runs_indivs.values(),
            keys=runs_indivs.keys(),
            names=[RunCol.ID, IndivCol.ID],
        )
        nodes = self.indivs[IndivCol.INSTANCE].apply(lambda i: [
                {NodeCol.ID: key} | value
                for key, value in i.graph.nodes.items()
            ]).explode()
        self.nodes = pd.DataFrame(
            nodes.tolist(),
            index=nodes.index,
        ).set_index(NodeCol.ID, append=True)
        self.nodes[NodeCol.TYPE] = self.nodes[NodeCol.INSTANCE].map(type)
        self.edges = pd.DataFrame(
                self.nodes[NodeCol.SUCCESSORS].explode().rename(EdgeCol.SOURCE_ID)
        )

    @cached_property
    def roots(self) -> pd.DataFrame:
        return self.nodes.groupby([RunCol.ID, IndivCol.ID]).first()

    @cached_property
    def leaves(self) -> pd.DataFrame:
        return self.nodes[~self.nodes.index.isin(self.edges.dropna().index)]


def last_gen(indivs: pd.DataFrame) -> int:
    return int(indivs[IndivCol.GEN].max())


def remove_incomplete_last_gen(
    pop: Population,
    pop_sizes: Sequence[int],
) -> Population:
    indivs = pop.evaluated_individuals
    last_gen_ = last_gen(indivs)
    if last_gen_ > len(pop_sizes) - 1:
        pop_size = pop_sizes[-1]
    else:
        pop_size = pop_sizes[last_gen_]
    last_gen_mask = indivs[IndivCol.GEN] == last_gen_
    if last_gen_mask.sum() >= pop_size:
        return pop
    indivs = indivs.loc[~last_gen_mask]
    pop.population = pop.population[:indivs.shape[0]]
    pop.evaluated_individuals = indivs
    return pop


def score_columns(indivs: pd.DataFrame) -> list[str]:
    non_score_cols = set(col.value for col in IndivCol)
    return [col for col in indivs.columns if col not in non_score_cols]


def pareto_mask(indivs: pd.DataFrame) -> pd.Series:
    return indivs[IndivCol.PARETO] == 1


def run_is_complete(dir: str | Path) -> bool:
    manager = cfg.parse_config(_io.load_params(dir), tpot.TPOTManager)
    try:
        indivs = _io.load_indivs(dir)
    except FileNotFoundError:
        return False
    return manager.indivs_is_complete(indivs)
