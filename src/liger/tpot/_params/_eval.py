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
from typing import Any, Literal


@dataclass(slots=True, frozen=True)
class EvalParams:
    cv: int = 10
    eval_time: float | None = 10.0
    threshold_evaluation_pruning: tuple[float, float] | None = None
    threshold_evaluation_scaling: float = 0.5
    selection_evaluation_pruning: tuple[float, float] | None = None
    selection_evaluation_scaling: float = 0.5
    min_history_threshold: int = 0
    budget_range: tuple[float, float] | None = None
    budget_scaling: float = 0.5
    generations_until_end_budget: int = 1
    validation_strategy: Literal["auto", "reshuffled", "split", "none"] = "none"
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.generations_until_end_budget < 1:
            raise ValueError("'generations_until_end_budget' must be at least 1")

    @property
    def gen_of_max_budget(self) -> int:
        if self.budget_range is None or self.budget_range[1] <= self.budget_range[0]:
            return 0
        # /// --- Corrects for 'tpot.utils.beta_interpolation()' edge case --- \\\
        return max(self.generations_until_end_budget - 1, 1)
        # \\\ ---------------------------------------------------------------- ///

    def tpot_kwargs(self) -> dict[str, Any]:
        return {
            "max_eval_time_mins": self.eval_time,
            "validation_strategy": self.validation_strategy,
            "validation_fraction": self.validation_fraction,
            "threshold_evaluation_pruning": self.threshold_evaluation_pruning,
            "threshold_evaluation_scaling": self.threshold_evaluation_scaling,
            "selection_evaluation_pruning": self.selection_evaluation_pruning,
            "selection_evaluation_scaling": self.selection_evaluation_scaling,
            "min_history_threshold": self.min_history_threshold,
            "budget_range": self.budget_range,
            "budget_scaling": self.budget_scaling,
            "generations_until_end_budget": self.generations_until_end_budget,
        }
