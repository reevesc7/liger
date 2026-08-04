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


from typing import Any, Iterable, Protocol
from dataclasses import dataclass, field
import inspect
import numpy as np
from sklearn.base import BaseEstimator


class ScorerCallable(Protocol):
    def __call__(self, estimator: BaseEstimator, X: Any, y_true: Any, /) -> Any: ...


class OtherCallable(Protocol):
    def __call__(self, estimator: BaseEstimator, /) -> Any: ...


ObjectiveCallable = ScorerCallable | OtherCallable


@dataclass(slots=True)
class Objective:
    func: str | ObjectiveCallable
    weight: float
    early_stop_tol: float | None = 0.001
    force_scorer: bool = False
    is_scorer_: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.force_scorer or isinstance(self.func, str):
            self.is_scorer_ = True
        else:
            self.is_scorer_ = self._func_is_scorer(self.func)

    @staticmethod
    def _func_is_scorer(func: ScorerCallable | OtherCallable) -> bool:
        signature = inspect.signature(func)
        params = signature.parameters.values()
        positional_capable_args = [
            param for param in params
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_capable_args) >= 3:
            return True
        if len(positional_capable_args) == 1:
            return False
        raise ValueError(
            f"Unable to infer objective type from signature {signature}; "
                "pass a value to 'is_scorer' at construction"
        )


@dataclass(slots=True)
class InverseObjectives:
    objectives: Iterable[Objective]
    scorers_: list[str | ObjectiveCallable] = field(init=False)
    scorers_weights_: list[float] = field(init=False)
    scorers_early_stop_tols_: list[float | None] = field(init=False)
    others_: list[str | ObjectiveCallable] = field(init=False)
    others_weights_: list[float] = field(init=False)
    others_early_stop_tols_: list[float | None] = field(init=False)
    signs_: np.ndarray = field(init=False)
    early_stop_tols_: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        scorers = [objctv for objctv in self.objectives if objctv.is_scorer_]
        self.scorers_ = [scorer.func for scorer in scorers]
        self.scorers_weights_ = [scorer.weight for scorer in scorers]
        self.scorers_early_stop_tols_ = [scorer.early_stop_tol for scorer in scorers]
        others = [objctv for objctv in self.objectives if not objctv.is_scorer_]
        self.others_ = [other.func for other in others]
        self.others_weights_ = [other.weight for other in others]
        self.others_early_stop_tols_ = [other.early_stop_tol for other in others]
        self.signs_ = np.sign(self.scorers_weights_ + self.others_weights_)
        self.early_stop_tols_ = np.array(
            self.scorers_early_stop_tols_ + self.others_early_stop_tols_
        )
