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


from typing import MutableSequence
from typing_extensions import Self
from pathlib import Path
import pandas as pd
#from .operations import vector_projection, vector_rejection


class Dataset:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"""Dataset:
    x: {self.x},
    y: {self.y}"""

    @staticmethod
    def _to_set(input: str | MutableSequence[str] | set[str]) -> set[str]:
        if isinstance(input, str):
            return {input,}
        if isinstance(input, MutableSequence):
            return {string for string in input}
        return input

    @staticmethod
    def _patterns_in(string: str, patterns: set[str]) -> bool:
        return any(pattern in string for pattern in patterns)

    @staticmethod
    def _filter_cols(cols: pd.Index, patterns: set[str]) -> pd.Index:
        return pd.Index(col for col in cols if Dataset._patterns_in(col, patterns))

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        feature_patterns: str | MutableSequence[str] | set[str],
        score_patterns: str | MutableSequence[str] | set[str],
    ) -> Self:
        feature_patterns = cls._to_set(feature_patterns)
        score_patterns = cls._to_set(score_patterns)
        return cls(
            df.filter(cls._filter_cols(df.columns, feature_patterns)),
            df.filter(cls._filter_cols(df.columns, score_patterns)),
        )

    @classmethod
    def from_csv(
        cls,
        file_path: str | Path,
        feature_patterns: str | MutableSequence[str] | set[str],
        score_patterns: str | MutableSequence[str] | set[str],
    ) -> Self:
        """
        Initialize a `Dataset` from a `csv` file.

        Parameters
        ----------
        `file_path` : `str`
            The path of the `csv` file to read.
        `feature_pattern` `str` or `MutableSequence[str]` or `set[str]`
            The pattern(s) to search for in column names to use for features.
            NOTE: If both `feature_pattern` and `score_pattern` are provided,
            only columns matching them will be loaded into memory.
        `score_pattern` `str` or `MutableSequence[str]` or `set[str]`
            The pattern(s) to search for in column names to use for scores.
            NOTE: If both `feature_pattern` and `score_pattern` are provided,
            only columns matching them will be loaded into memory.

        Returns
        -------
        `dataset` : `Dataset`
            A dataset with data matching the csv file contents,
            potentially filtered (see above).
        """
        feature_patterns = cls._to_set(feature_patterns)
        score_patterns = cls._to_set(score_patterns)
        return cls(
            pd.read_csv(file_path, usecols=lambda col: cls._patterns_in(col, feature_patterns)),
            pd.read_csv(file_path, usecols=lambda col: cls._patterns_in(col, score_patterns)),
        )

    def to_csv(self, filename: str | Path) -> None:
        """Save this dataset as a `csv` file.

        ...
        """
        pd.concat((self.x, self.y), axis=1).to_csv(filename, index=False)

    #def analyze_manifold(self, point_a: ArrayLike, point_b: ArrayLike) -> pd.DataFrame:
    #    manifold = pd.DataFrame({key: np.zeros(self.y.shape[0]) for key in ["y", "alpha", "dist"]})
    #    ab_vector = np.subtract(point_b, point_a)
    #    manifold["y"] = self.y
    #    for i, point_p in enumerate(self.X):
    #        ap_vector = np.subtract(point_p, point_a)
    #        manifold["alpha"][i] = (vector_projection(ab_vector, ap_vector)*8)+1
    #        manifold["dist"][i] = vector_rejection(ab_vector, ap_vector)
    #    return manifold

