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


# Calculates a point along the line segment between point1 and point2, positioned in a range [0,1] along the segment.
def interpolate_points(point1: ArrayLike, point2: ArrayLike, alpha: float) -> np.ndarray:
    point1 = np.array(point1)
    point2 = np.array(point2)
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha should be between 0 and 1.")
    interpolated_point = point1 + alpha * (point2 - point1)
    return interpolated_point


# Calculates the projection of AP onto AB as a proportion of the magnitude of AB.
def vector_projection(ABvector: ArrayLike, APvector: ArrayLike) -> float:
    return np.dot(APvector, ABvector) / np.dot(ABvector, ABvector)


# Calculates the rejection of AP from AB.
def vector_rejection(ABvector: ArrayLike, APvector: ArrayLike) -> float:
    AP0vector = vector_projection(ABvector, APvector) * ABvector
    return np.linalg.norm(APvector - AP0vector)

def compute_R2(data: list[ArrayLike], equations: list) -> list:
    actual_y = data[1]
    return [np.corrcoef(actual_y, [equation(x) for x in data[0]])[0,1]**2 for equation in equations]

