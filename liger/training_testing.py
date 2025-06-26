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
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics._scorer import _Scorer
from .dataset import Dataset


# Takes a model, data in Dataset format, indices of data to train on. Returns trained model.
def train_model(model, data: Dataset, train_indices: np.ndarray):
    model.fit(data.x[train_indices], data.y[train_indices])
    return model


# Takes a model, data in Dataset format, indices of data to tset on. Returns performance measures
def evaluate_model(
    model,
    data: Dataset,
    test_indices: np.ndarray
) -> tuple[float, float]:
    model_predictions = model.predict(data.x[test_indices])
    mse = float(mean_squared_error(model_predictions, data.y[test_indices]))
    r2 = float(r2_score(model_predictions, data.y[test_indices]))
    return mse, r2


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
        model_clone = clone(model)
        model_clone.fit(data.x[train_indices], data.y[train_indices])
        predicted.append(dict(zip(test_indices.tolist(), model_clone.predict(data.x[test_indices]).tolist())))
        fold_scores[fold][scorer_indices] = [
            scorers[index]._score_func(data.y[test_indices], list(predicted[-1].values()))
            for index in scorer_indices
        ]
    scores = np.average(fold_scores, axis=0)
    scores[objective_indices] = [scorers[index](model) for index in objective_indices]
    return predicted, scores.tolist()

