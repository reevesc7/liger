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


def pca(data: pd.DataFrame, n_components: int | None = None) -> tuple[PCA, pd.DataFrame]:
    pca = PCA(n_components)
    reduced_data = pca.fit_transform(data)
    return pca, pd.DataFrame(
        reduced_data,
        columns=pd.Index(f"pc_{pc}" for pc in pca.n_components_)
    )

