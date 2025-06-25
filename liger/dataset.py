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


import ast
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from .operations import vector_projection, vector_rejection


class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of entries")
        if len(x.shape) > 2 or len(y.shape) > 2:
            raise ValueError("x and y must be 1- or 2-dimensional")
        self.X = x
        self.y = y

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        feature_keys: list[str],
        score_keys: list[str],
    ) -> 'Dataset':
        """Initialize a `Dataset` from a `pandas.DataFrame`.

        Parameters
        ----------
        `df` : `pandas.DataFrame`
            The data to be incorporated into the `Dataset`.
        `feature_keys` : `list[str]`
            The column keys of inputs to models.
        `score_keys` : `list[str]`
            The column keys of sample scores.

        Returns
        -------
        `dataset` : `Dataset`
            A dataset containing 1D or 2D numpy arrays for `X` and `y`.
        """
        df = cls._parse_df_strs(df)
        df = cls._concatenate_df_cols(df, feature_keys, score_keys)
        return Dataset(np.array(df["X"].to_list()), np.array(df["y"].to_list()))

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        feature_keys: list[str],
        score_keys: list[str],
    ) -> "Dataset":
        """Initialize a `Dataset` from a `csv` file.

        Parameters
        ----------
        `file_path` : `str`
            The path of the `csv` file to read.
        `feature_keys` : `list[str]`
            The column keys of inputs to models.
        `score_keys` : `list[str]`
            The column keys of sample scores.

        Returns
        -------
        `dataset` : `Dataset`
            A dataset containing 1D or 2D numpy arrays for `X` and `y`.
        """
        try:
            df = pd.read_csv(file_path, usecols=np.array(feature_keys + score_keys))
        except ValueError:
            print("Key in config `feature_keys` or `score_key` not found in dataset")
            raise
        dataset = Dataset.from_df(df, feature_keys, score_keys)
        return dataset

    def __str__(self):
        return f"X: {self.X},\ny: {self.y}"

    @staticmethod
    def _parse_df_strs(df: pd.DataFrame) -> pd.DataFrame:
        """Interpret string representations of data types in a dataframe.
        Can convert `int`, `float`, `list`, `tuple`, `dict`, `set`.

        Parameters
        ----------
        `df` : `pandas.Dataframe`
            The dataframe to be parsed.

        Returns
        -------
        `parsed_df` : `pandas.DataFrame`
            The original dataframe with all parseable columns interpreted.
        """
        data = {}
        for column in df.columns:
            try:
                data[column] = df[column].apply(lambda x: np.array(ast.literal_eval(x)))
            except ValueError as e:
                if "malformed" in str(e):
                    data[column] = df[column]
                else:
                    raise
        return pd.DataFrame(data)

    @staticmethod
    def _concat_cols(arrays: list) -> np.ndarray:
        for index, array in enumerate(arrays):
            if len(array.shape) == 0:
                arrays[index] = np.array([array])
        return np.concatenate(arrays)

    @staticmethod
    def _concatenate_df_cols(
        df: pd.DataFrame,
        feature_keys: list[str],
        score_keys: list[str],
    ) -> pd.DataFrame:
        if len(feature_keys) > 1:
            x = df.apply(
                lambda row: Dataset._concat_cols([np.array(row[key]) for key in feature_keys]),
                axis=1
            )
        else:
            x = df[feature_keys[0]]
        if len(score_keys) > 1:
            y = df.apply(
                lambda row: Dataset._concat_cols([np.array(row[key]) for key in score_keys]),
                axis=1
            )
        else:
            y = df[score_keys[0]]
        return pd.DataFrame({"X": x, "y": y})

    @staticmethod
    def _2darray2string(a: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda row: np.array2string(
            row,
            max_line_width=10000000,
            separator=", ",
            threshold=10000000,
        ), axis=1, arr=a)

    @property
    def X_strings(self) -> np.ndarray:
        """Full string representations of each element of X.
        """
        if len(self.X.shape) == 1:
            return self.X
        else:
            return self._2darray2string(self.X)

    @property
    def y_strings(self) -> np.ndarray:
        """Full string representations of each element of y.
        """
        if len(self.y.shape) == 1:
            return self.y
        else:
            return self._2darray2string(self.y)

    def to_df(self) -> pd.DataFrame:
        """Convert this dataset to a dataframe.

        Returns
        -------
        `df` : `pandas.DataFrame`
            A dataframe representation of this dataset, with columns "X" and "y",
            filled with string representations of `X` and `y` array elements.
        """
        return pd.DataFrame({"X": self.X_strings, "y": self.y_strings})

    def to_csv(self, filename: str) -> None:
        """Save this dataset as a `csv` file.

        Arrays in `X` and `y` are converted to full string representations
        """
        self.to_df().to_csv(filename, index=False)

    def analyze_manifold(self, point_a: ArrayLike, point_b: ArrayLike) -> pd.DataFrame:
        manifold = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in ["y", "alpha", "dist"]})
        ab_vector = np.subtract(point_b, point_a)
        manifold["y"] = self.y
        for i, point_p in enumerate(self.X):
            ap_vector = np.subtract(point_p, point_a)
            manifold["alpha"][i] = (vector_projection(ab_vector, ap_vector)*8)+1
            manifold["dist"][i] = vector_rejection(ab_vector, ap_vector)
        return manifold

