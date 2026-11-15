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


from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from numpy.typing import ArrayLike
from tpot.selectors import survival_select_NSGA2, tournament_selection_dominated
from tpot.utils import beta_interpolation


@dataclass(slots=True, frozen=True)
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

    def __post_init__(self) -> None:
        if self.generations_until_end_population < 1:
            raise ValueError("'generations_until_end_population' must be at least 1")

    @property
    def gen_of_end_pop_size(self) -> int:
        # /// --- Corrects for 'beta_interpolation()' edge case --- \\\
        return max(self.generations_until_end_population - 1, 1)
        # \\\ ----------------------------------------------------- ///

    @property
    def gens_pop_sizes(self) -> list[int]:
        start_size = self.initial_population_size
        end_size = self.population_size
        if start_size is None or start_size == end_size:
            return [end_size]
        per_gen_pop_size: list[np.float64] = beta_interpolation(
            start=start_size,
            end=end_size,
            scale=self.population_scaling, # type: ignore
            n=self.gen_of_end_pop_size + 1,
        )
        return np.round(per_gen_pop_size).astype(int).tolist()

    def gen_pop_size(self, gen: int) -> int:
        return self.gens_pop_sizes[min(gen, self.gen_of_end_pop_size)]

    def tpot_kwargs(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "initial_population_size": self.initial_population_size,
            "population_scaling": self.population_scaling,
            "generations_until_end_population": self.generations_until_end_population,
            "survival_percentage": self.survival_percentage,
            "crossover_probability": self.crossover_probability,
            "mutate_probability": self.mutate_probability,
            "mutate_then_crossover_probability": self.mutate_then_crossover_probability,
            "crossover_then_mutate_probability": self.crossover_then_mutate_probability,
            "survival_selector": self.survival_selector,
            "parent_selector": self.parent_selector,
        }
