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


from typing import Sequence
import pandas as pd
from tpot import Population
from ._schema import Col


def last_gen(indivs: pd.DataFrame) -> int:
    return int(indivs[Col.GEN].max())


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
    last_gen_mask = indivs[Col.GEN] == last_gen_
    if last_gen_mask.sum() >= pop_size:
        return pop
    indivs = indivs.loc[~last_gen_mask]
    pop.population = pop.population[:indivs.shape[0]]
    pop.evaluated_individuals = indivs
    return pop


def score_columns(indivs: pd.DataFrame) -> list[str]:
    non_score_cols = set(col.value for col in Col)
    return [col for col in indivs.columns if col not in non_score_cols]
