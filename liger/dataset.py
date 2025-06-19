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
    def __init__(self, n_entries: int, X_len: int, y_len: int = 1):
        self.X = np.array(np.zeros((n_entries, X_len)))
        if y_len == 1:
            self.y = np.array(np.zeros(n_entries))
        else:
            self.y = np.array(np.zeros((n_entries, y_len)))

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        feature_keys: list[str] | list[int],
        score_key: str | int,
    ) -> 'Dataset':
        df = cls._parse_df_lists(df)
        df = cls._concatenate_df_cols(df, feature_keys, score_key)
        X_len = len(df["X"][0])
        try:
            y_len = len(df["y"][0])
        except TypeError:
            y_len = 1
        dataset = Dataset(df.shape[0], X_len, y_len)
        dataset.X = np.array(df["X"].tolist())
        dataset.y = np.array(df["y"].tolist())
        return dataset

    @classmethod
    def from_csv(
        cls,
        filename: str,
        feature_keys: list[str] | list[int],
        score_key: str | int,
    ) -> "Dataset":
        try:
            df = pd.read_csv(filename, usecols=np.array(feature_keys + [score_key]))
        except ValueError:
            print("Key in config `feature_keys` or `score_key` not found in dataset")
            raise
        dataset = Dataset.from_df(df, feature_keys, score_key)
        return dataset

    def __str__(self):
        return f"X: {self.X},\ny: {self.y}"

    @staticmethod
    def _parse_df_lists(df: pd.DataFrame) -> pd.DataFrame:
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
    def _concatenate_df_cols(
        df: pd.DataFrame,
        feature_keys: list[str] | list[int],
        score_key: str | int
    ) -> pd.DataFrame:
        print(df)
        return pd.DataFrame({
            "X": df.apply(lambda row: np.concatenate([row[key] for key in feature_keys]), axis=1),
            "y": df[score_key],
        })

    def flatten(self) -> pd.DataFrame:
        df = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in ["X", "y"]})
        df["X"] = np.apply_along_axis(lambda row: np.array2string(row, max_line_width=10000000, separator=", ", threshold=10000000), axis=1, arr=self.X)
        df["y"] = self.y
        return df

    def to_csv(self, filename):
        self.flatten().to_csv(filename, index=False)

    def analyze_manifold(self, point_a: ArrayLike, point_b: ArrayLike) -> pd.DataFrame:
        manifold = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in ["y", "alpha", "dist"]})
        ab_vector = np.subtract(point_b, point_a)
        manifold["y"] = self.y
        for i, point_p in enumerate(self.X):
            ap_vector = np.subtract(point_p, point_a)
            manifold["alpha"][i] = (vector_projection(ab_vector, ap_vector)*8)+1
            manifold["dist"][i] = vector_rejection(ab_vector, ap_vector)
        return manifold

