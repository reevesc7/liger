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


from typing import Callable, Literal
from dataclasses import dataclass, field
from random import randint
import numpy as np
from numpy.typing import ArrayLike
from tpot.selectors import survival_select_NSGA2, tournament_selection_dominated
from tpot.utils import beta_interpolation


@dataclass(slots=True)
class EvolutionParams:
    population_size: int = 50
    initial_population_size: int | None = None
    population_scaling: float = 0.5
    generations_until_end_population: int = 1
    survival_selector: Callable[[ArrayLike, int, int], ArrayLike] = (
        survival_select_NSGA2
    )
    survival_percentage: float = 1.0
    parent_selector: Callable[[ArrayLike, int, int, int], ArrayLike] = (
        tournament_selection_dominated
    )
    crossover_probability: float = 0.2
    mutate_probability: float = 0.7
    mutate_then_crossover_probability: float = 0.05
    crossover_then_mutate_probability: float = 0.05
    random_state: Literal["auto"] | int | None = "auto"
    random_state_: int | None = field(init=False)
    gen_of_end_pop_size_: int = field(init=False)
    gens_pop_sizes_: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.random_state == "auto":
            self.random_state = self.random_state_ = randint(0, 2 ** 32 - 1)
        else:
            self.random_state_ = self.random_state
        # /// --- Corrects for 'beta_interpolation()' edge case --- \\\
        if self.generations_until_end_population <= 1:
            self.gen_of_end_pop_size_ = 2
        else:
            self.gen_of_end_pop_size_ = self.generations_until_end_population
        # \\\ ----------------------------------------------------- ///
        self.gens_pop_sizes_ = self._gens_pop_sizes()

    def _gens_pop_sizes(self) -> list[int]:
        start_size = self.initial_population_size
        end_size = self.population_size
        if start_size is None or start_size == end_size:
            return [end_size]
        gen_of_end_size = self.generations_until_end_population
        # /// --- Corrects for 'beta_interpolation()' edge case --- \\\
        if gen_of_end_size <= 1:
            gen_of_end_size = 2
        # \\\ ----------------------------------------------------- ///
        # if gen >= gen_of_end_size:
        #     return end_size
        per_gen_pop_size: list[np.float64] = beta_interpolation(
            start=start_size,
            end=end_size,
            scale=self.evolution_params.population_scaling, # type: ignore
            n=self.gen_of_end_pop_size_,
        )
        return np.round(per_gen_pop_size).astype(int).tolist()

    def gen_pop_size(self, gen: int) -> int:
        if gen > self.gen_of_end_pop_size_:
            gen = self.gen_of_end_pop_size_
        return self.gens_pop_sizes_[gen]
