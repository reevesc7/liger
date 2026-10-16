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
from numpy.typing import ArrayLike
import pandas as pd


def linear_fit_array(
    data: ArrayLike,
    point1: ArrayLike,
    point2: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Squash the data onto a line segment between two points
    and report the proportional distance of each point along
    the line segment (alpha) and the distance of each
    point from the line segment (deviation).
    #
    ...
    """
    data = np.asarray(data, dtype=float)
    point1 = np.asarray(point1, dtype=float)
    target = np.asarray(point2, dtype=float) - point1
    vectors = data - point1
    alphas = (vectors @ target) / (target @ target)
    residuals = vectors - alphas[..., None] * target
    deviations = np.linalg.norm(residuals, axis=-1)
    return alphas, deviations


def linear_fit(
    data: pd.DataFrame,
    point1: ArrayLike,
    point2: ArrayLike,
) -> pd.DataFrame:
    """Squash the data onto a line segment between two points
    and report the proportional distance of each point along
    the line segment (alpha) and the distance of each
    point from the line segment (deviation).
    #
    ...
    """
    alphas, deviations = linear_fit_array(data.to_numpy(), point1, point2)
    return pd.DataFrame(
        {"alpha": alphas, "deviation": deviations},
        index=data.index,
    )
