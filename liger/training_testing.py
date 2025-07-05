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


from typing import Any
from types import FunctionType
from importlib import import_module
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics._scorer import _Scorer
from .dataset import Dataset


def init_scorers(param_scorers: list[str]) -> list[str | FunctionType]:
    scorers: list[str | FunctionType] = []
    for param_scorer in param_scorers:
        if "." not in param_scorer:
            scorers.append(param_scorer)
            continue
        split_scorer = param_scorer.rsplit(".", 1)
        scorers.append(getattr(import_module(split_scorer[0]), split_scorer[1]))
    return scorers


def _separate_objectives(scorers: list[Any]) -> tuple[list[int], list[int]]:
    scorer_indices = [
        index
        for index, element in enumerate(scorers)
        if isinstance(element, _Scorer)
    ]
    objective_indices = [
        index
        for index in range(len(scorers))
        if index not in scorer_indices
    ]
    return scorer_indices, objective_indices


# Returns a model's predictions across all training instances of a KFold cross validation
def kfold_predict(
    model,
    kfold: KFold,
    scorers: list[Any],
    data: Dataset
) -> tuple[list[dict[int, Any]], list[float | list[float]]]:
    predicted: list[dict[int, Any]] = []
    fold_scores = np.zeros((kfold.get_n_splits(), len(scorers)))
    scorer_indices, objective_indices = _separate_objectives(scorers)
    for fold, [train_indices, test_indices] in enumerate(kfold.split(data.x)):
        model.fit(data.x.iloc[train_indices], data.y.iloc[train_indices])
        predicted.append(dict(zip(
            test_indices.tolist(),
            model.predict(data.x.iloc[test_indices]).tolist()
        )))
        fold_scores[fold][scorer_indices] = [
            scorers[index]._score_func(data.y.iloc[test_indices], list(predicted[-1].values()))
            for index in scorer_indices
        ]
    scores = np.average(fold_scores, axis=0)
    scores[objective_indices] = [scorers[index](model) for index in objective_indices]
    return predicted, scores.tolist()

