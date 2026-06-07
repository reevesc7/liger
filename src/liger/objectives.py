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


from sklearn.metrics import make_scorer
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import wasserstein_distance
from liger import probabilities as prb


def score_dummy(*args, **kwargs) -> float:
    _ = args, kwargs
    return 0.0


def score_msle_2d(y_true: ArrayLike, y_pred: ArrayLike) -> np.floating:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    d1_true, d2_true = y_true[:, 0], y_true[:, 1]
    d1_pred, d2_pred = y_pred[:, 0], y_pred[:, 1]
    return score_msle(d1_true, d1_pred) + score_msle(d2_true, d2_pred)


def score_msle(y_true: ArrayLike, y_pred: ArrayLike) -> np.floating:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_clip = np.clip(y_true, 1e-6, None)
    y_pred_clip = np.clip(y_pred, 1e-6, None)
    return np.mean(np.log(y_pred_clip) - np.log(y_true_clip))**2


def score_nll_mean_stdev(y_true: ArrayLike, y_pred: ArrayLike) -> np.floating:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_pred, stdev_pred = y_pred[:, 0], y_pred[:, 1]
    return np.mean(np.log(stdev_pred) + (y_true - mean_pred)**2 / (2 * stdev_pred**2))


def score_test(y_true: ArrayLike, y_pred: ArrayLike) -> np.floating:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = y_true[:, 0]
    mean_pred, stdev_pred = y_pred[:, 0], y_pred[:, 1]
    a = (mean_pred - mean_true)**2
    b = mean_true**2 * np.log(mean_true**2 / (mean_pred**2 + stdev_pred**2))
    c = mean_pred**2
    d = stdev_pred**2
    e = mean_true**2
    return np.mean(a + b + c + d - e)


# def soft_brier(y_true_pmf: ArrayLike, y_pred_proba: ArrayLike) -> np.floating[Any]:
    # y_true = np.asarray(y_true)
    # y_pred = np.asarray(y_pred)
#     return np.mean(np.sum((y_pred_proba - y_true_pmf) ** 2, axis=1))


def _row_wasserstein(
    pmf_true: ArrayLike,
    pmf_pred: ArrayLike,
    norm: bool = False,
) -> float:
    pmf_true = np.asarray(pmf_true)
    pmf_pred = np.asarray(pmf_pred)
    if pmf_true.ndim != 1:
        raise ValueError(f"Expected 1D array-like inputs, but pmf_true is shape {pmf_true.shape}")
    if pmf_pred.ndim != 1:
        raise ValueError(f"Expected 1D array-like inputs, but pmf_pred is shape {pmf_pred.shape}")
    if norm:
        x_vals = [support / (pmf_true.size - 1) for support in range(pmf_true.size)]
    else:
        x_vals = range(pmf_true.size)
    return wasserstein_distance(x_vals, x_vals, pmf_true, pmf_pred)


def score_wasserstein(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    norm: bool = False
) -> np.floating:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    distances = np.full(y_true.shape[0], 0.0)
    for i in range(len(distances)):
        distances[i] = _row_wasserstein(y_true[i], y_pred[i], norm)
    return np.mean(distances)


def score_softmax_wasserstein(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    norm: bool = False,
    temperature: float = 1.0,
) -> np.floating:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pmf_true = prb.softmax(y_true, temperature)
    pmf_pred = prb.softmax(y_pred, temperature)
    return score_wasserstein(pmf_true, pmf_pred, norm)


# Define scorers
dummy = make_scorer(
    score_dummy,
    response_method="predict",
    greater_is_better=False,
)
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
# neg_soft_brier = make_scorer(
#     soft_brier,
#     response_method="predict_proba",
#     greater_is_better=False,
# )
neg_wasserstein = make_scorer(
    score_wasserstein,
    response_method="predict",
    greater_is_better=False,
)
neg_softmax_wasserstein = make_scorer(
    score_softmax_wasserstein,
    response_method="predict",
    greater_is_better=False,
    **{
        "temperature": 1.0,
    },
)
neg_norm_wasserstein = make_scorer(
    score_wasserstein,
    response_method="predict",
    greater_is_better=False,
    **{
        "norm": True,
    },
)
neg_softmax_norm_wasserstein = make_scorer(
    score_softmax_wasserstein,
    response_method="predict",
    greater_is_better=False,
    **{
        "norm": True,
        "temperature": 1.0,
    },
)

