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


import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy import special


def linear_projection(
    data: ArrayLike,
    point1: ArrayLike,
    point2: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project the data onto a line defined by two points
    and report the distance of each point along the line as
    a proportion of position from the first point to the second (alpha)
    and the distance of each point from the line (deviation).

    Parameters
    ----------
    data : ArrayLike
        ND array of 1D points.
    point1 : ArrayLike
        1D array representing the 0.0 coordinate of the line.
    point2 : ArrayLike
        1D array representing the 1.0 coordinate of the line.

    Returns
    -------
    alphas, deviations : tuple[NDArray[float64], NDArray[float64]]
        2 (N - 1)D arrays. Alphas represent the proportional position of
        each point along the line, from point1 to point2. E.g., a point whose
        closest point on the line was point1 would have alpha == 1.0.
        Alphas may be outside of the range [0.0, 1.0].
        Deviations represent how far each point originally was from the line.
    """
    data = np.asarray(data, dtype=np.float64)
    point1 = np.asarray(point1, dtype=np.float64)
    target = np.asarray(point2, dtype=np.float64) - point1
    if (target == 0.0).all():
        raise ValueError("point1 and point2 must be different points")
    vectors = data - point1
    alphas = (vectors @ target) / (target @ target)
    residuals = vectors - alphas[..., None] * target
    deviations = np.linalg.norm(residuals, axis=-1)
    return alphas, deviations


def passthrough(x_vals: ArrayLike) -> np.ndarray:
    return np.asarray(x_vals)


def softmax(
    logprobs: ArrayLike,
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    logprobs = np.asarray(logprobs, dtype=np.float64)
    temperature = max(temperature, 0.0)
    if temperature == 0.0:
        return (logprobs == logprobs.max(axis=-1, keepdims=True)).astype(float)
    return special.softmax(logprobs / temperature, axis=-1)


def pmf_mode(x_vals: ArrayLike, masses: ArrayLike) -> NDArray:
    x_vals = np.asarray(x_vals)
    masses = np.asarray(masses)
    mode_indices = np.argmax(masses, axis=-1, keepdims=True)
    return np.take_along_axis(x_vals, mode_indices, axis=-1)[..., 0]


def pmf_mean(x_vals: ArrayLike, masses: ArrayLike) -> NDArray[np.float64]:
    x_vals = np.asarray(x_vals, dtype=np.float64)
    masses = np.asarray(masses)
    return np.average(x_vals, axis=-1, weights=masses)


def pmf_variance(x_vals: ArrayLike, masses: ArrayLike) -> NDArray[np.float64]:
    x_vals = np.asarray(x_vals, dtype=np.float64)
    masses = np.asarray(masses)
    return np.average(
        (x_vals.T - pmf_mean(x_vals, masses)).T ** 2,
        axis=-1,
        weights=masses,
    )


def pmf_std_dev(x_vals: ArrayLike, masses: ArrayLike) -> NDArray[np.float64]:
    return np.sqrt(pmf_variance(x_vals, masses))


def mean_center(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x, axis, keepdims=True)


def apply_passthrough(
    logprobs: pd.DataFrame,
) -> pd.DataFrame:
    return logprobs


def _strip_prefix(
    columns: pd.Index,
    prefix: str,
) -> pd.Index:
    return pd.Index([
        str(col).removeprefix(prefix)
        for col in columns
    ])


def apply_softmax(
    logprobs: pd.DataFrame,
    temperature: float = 1.0,
    strip_prefix: str = "logprob_",
    add_prefix: str = "prob_",
) -> pd.DataFrame:
    x_vals = _strip_prefix(logprobs.columns, strip_prefix)
    return pd.DataFrame(
        softmax(logprobs.to_numpy(), temperature),
        index=logprobs.index,
        columns=pd.Index([f"{add_prefix}{x_val}" for x_val in x_vals]),
    )


def _format_x_vals(df: pd.DataFrame, strip_prefix: str) -> np.ndarray:
    return np.broadcast_to(
        _strip_prefix(df.columns, strip_prefix).to_numpy(dtype=float),
        df.shape,
    )


def apply_logprobs_mode(
    logprobs: pd.DataFrame,
    temperature: float = 1.0,
    strip_prefix: str = "logprob_",
) -> pd.Series:
    masses = apply_softmax(logprobs, temperature, strip_prefix)
    x_vals = _format_x_vals(logprobs, strip_prefix)
    return pd.Series(pmf_mode(x_vals, masses), name="mode")


def apply_logprobs_mean(
    logprobs: pd.DataFrame,
    temperature: float = 1.0,
    strip_prefix: str = "logprob_",
) -> pd.Series:
    masses = apply_softmax(logprobs, temperature, strip_prefix)
    x_vals = _format_x_vals(logprobs, strip_prefix)
    return pd.Series(pmf_mean(x_vals, masses), name="mean")


def apply_logprobs_variance(
    logprobs: pd.DataFrame,
    temperature: float = 1.0,
    strip_prefix: str = "logprob_",
) -> pd.Series:
    masses = apply_softmax(logprobs, temperature, strip_prefix)
    x_vals = _format_x_vals(logprobs, strip_prefix)
    return pd.Series(pmf_variance(x_vals, masses), name="variance")


def apply_logprobs_std_dev(
    logprobs: pd.DataFrame,
    temperature: float = 1.0,
    strip_prefix: str = "logprob_",
) -> pd.Series:
    masses = apply_softmax(logprobs, temperature, strip_prefix)
    x_vals = _format_x_vals(logprobs, strip_prefix)
    return pd.Series(pmf_std_dev(x_vals, masses), name="std_dev")


def apply_mean_center(
    x: pd.DataFrame,
    axis: int | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(mean_center(x, axis), index=x.index, columns=x.columns)
