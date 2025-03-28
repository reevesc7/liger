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
from sklearn.metrics import make_scorer
import numpy as np
from numpy.typing import NDArray


def score_msle_2d(y_true: NDArray, y_pred: NDArray) -> np.floating[Any]:
    d1_true, d2_true = y_true[:, 0], y_true[:, 1]
    d1_pred, d2_pred = y_pred[:, 0], y_pred[:, 1]
    return score_msle(d1_true, d1_pred) + score_msle(d2_true, d2_pred)


def score_msle(y_true: NDArray, y_pred: NDArray) -> np.floating[Any]:
    y_true_clip = np.clip(y_true, 1e-6, None)
    y_pred_clip = np.clip(y_pred, 1e-6, None)
    return np.mean(np.log(y_pred_clip) - np.log(y_true_clip))**2


def score_nll_mean_stdev(y_true: NDArray, y_pred: NDArray) -> np.floating[Any]:
    mean_pred, stdev_pred = y_pred[:, 0], y_pred[:, 1]
    return np.mean(np.log(stdev_pred) + (y_true - mean_pred)**2 / (2 * stdev_pred**2))


def score_test(y_true: NDArray, y_pred: NDArray) -> np.floating[Any]:
    mean_true = y_true[:, 0]
    mean_pred, stdev_pred = y_pred[:, 0], y_pred[:, 1]
    a = (mean_pred - mean_true)**2
    b = mean_true**2 * np.log(mean_true**2 / (mean_pred**2 + stdev_pred**2))
    c = mean_pred**2
    d = stdev_pred**2
    e = mean_true**2
    return np.mean(a + b + c + d - e)


# Define scorers
neg_msle_2d = make_scorer(
    score_msle_2d,
    response_method="predict",
    greater_is_better=False,
)
neg_test = make_scorer(
    score_test,
    response_method="predict",
    greater_is_better=False,
)

