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
from numpy.typing import ArrayLike
from tpot.selectors import survival_select_NSGA2, tournament_selection_dominated


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

    def __post_init__(self) -> None:
        if self.random_state == "auto":
            self.random_state = self.random_state_ = randint(0, 2 ** 32 - 1)
        else:
            self.random_state_ = self.random_state
