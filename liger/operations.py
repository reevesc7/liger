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


# Calculates the projection of AP onto AB as a proportion of the magnitude of AB.
def vector_projection(ab_vector: ArrayLike, ap_vector: ArrayLike) -> float:
    return np.dot(ap_vector, ab_vector) / np.dot(ab_vector, ab_vector)


# Calculates the rejection of AP from AB.
def vector_rejection(ab_vector: ArrayLike, ap_vector: ArrayLike) -> float:
    ab_vector = np.array(ab_vector)
    ap_vector = np.array(ap_vector)
    ap_0_vector = vector_projection(ab_vector, ap_vector) * ab_vector
    return float(np.linalg.norm(ap_vector - ap_0_vector))

