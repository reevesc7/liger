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


from typing import Any, Callable
from dataclasses import dataclass
from importlib import import_module
from statistics import mean
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from .dataset import Dataset


@dataclass(slots=True)
class KFoldScores:
    predictions: list[dict[int, Any]]
    samples_scores: list[list[dict[int, float]]]
    fold_scores: list[list[float]]


def init_objects(param_objects: list[str]) -> list[str | Callable]:
    objects: list[str | Callable] = []
    for param_object in param_objects:
        if "." not in param_object:
            objects.append(param_object)
            continue
        split_scorer = param_object.rsplit(".", 1)
        objects.append(getattr(import_module(split_scorer[0]), split_scorer[1]))
    return objects


# Returns a model's predictions across all training instances of a KFold cross validation
def kfold_scores(
    model,
    kfold: KFold | StratifiedKFold,
    scorers: list[Any],
    data: Dataset
) -> KFoldScores:
    predicted: list[dict[int, Any]] = []
    fold_scores: list[list[float]] = [[] for _ in range(len(scorers))]
    samples_scores: list[list[dict[int, float]]] = [[] for _ in range(len(scorers))]
    for [train_indices, test_indices] in kfold.split(data.x, data.y):
        model_clone = clone(model)
        model_clone.fit(data.x.iloc[train_indices], data.y.iloc[train_indices])
        fold_predicted = [
            prediction[0] if isinstance(prediction, list) and len(prediction) == 1 else prediction
            for prediction in model_clone.predict(data.x.iloc[test_indices]).tolist()
        ]
        predicted.append(dict(zip(test_indices.tolist(), fold_predicted)))
        for scorer_index, scorer in enumerate(scorers):
            samples_scorer_scores = {
                int(test_index): scorer(
                    model_clone,
                    data.x.iloc[test_index].to_frame().T,
                    data.y.iloc[test_index].to_frame().T,
                )
                for test_index in test_indices
            }
            samples_scores[scorer_index].append(samples_scorer_scores)
            fold_scores[scorer_index].append(mean(samples_scorer_scores.values()))
    return KFoldScores(predicted, samples_scores, fold_scores)


def other_objective_scores(model, other_objectives: list[Any]) -> list[Any]:
    return [objective(model) for objective in other_objectives]

