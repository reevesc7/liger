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
from sklearn.decomposition import PCA


def _projection_coeff(vector: ArrayLike, target: ArrayLike) -> float:
    target = np.asarray(target)
    vector = np.asarray(vector)
    return np.dot(vector, target) / np.dot(target, target)


def _rejection(vector: ArrayLike, target: ArrayLike) -> float:
    target = np.asarray(target)
    vector = np.asarray(vector)
    projection = _projection_coeff(vector, target) * target
    return float(np.linalg.norm(vector - projection))


def linear_fit(data: pd.DataFrame, point1: ArrayLike, point2: ArrayLike) -> pd.DataFrame:
    """Squash the data onto a line segment between two points and report the proportional
    distance of each point along the line segment (alpha) and the distance of each
    point from the line segment (deviation).

    ...
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    target = point2 - point1
    x = data.apply(
        lambda row: pd.Series((
            _projection_coeff(row - point1, target),
            _rejection(row - point1, target)
        ), index=pd.Index(("alpha", "deviation"))),
        axis=1,
    )
    if not isinstance(x, pd.DataFrame):
        raise TypeError("This error should not happen. x is not a DataFrame")
    return x


def pca(data: pd.DataFrame, n_components: int | None = None) -> tuple[PCA, pd.DataFrame]:
    pca = PCA(n_components)
    reduced_data = pca.fit_transform(data)
    return pca, pd.DataFrame(
        reduced_data,
        columns=pd.Index(f"pc_{pc}" for pc in pca.n_components_)
    )

