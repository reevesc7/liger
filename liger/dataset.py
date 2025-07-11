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


from typing import overload, MutableSequence
from typing_extensions import Self
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


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

    @classmethod
    def random_linear(
        cls,
        n_entries: int,
        point1: ArrayLike,
        point2: ArrayLike,
        noise: float = 0.0,
        random_state: int | None = None,
    ) -> Self:
        """Generate a dataset of random points on the line segment between two given
        points, with scores corresponding to the distance of each point along that
        line segment and noise applied to each point.

        Parameters
        ----------
        `n_entries` : `int`
            The number of samples to include in the dataset.
        `point1` : `ArrayLike`
            An array of scalar values.
        `point2` : `ArrayLike`
            An array of scalar values. Should be the same shape as `point1`.
        `noise` : `float`, default 0.0
            How much each dimension of each sample can deviate, positive or negative,
            from the sample's "true" value in that dimension.
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        rng = np.random.default_rng(random_state)
        y = pd.DataFrame(rng.random(n_entries), columns=pd.Index(["y"]))
        x = pd.DataFrame(
            np.stack([cls.interpolated_point(point1, point2, alpha) for alpha in y.iloc[:, 0]]),
            columns=pd.Index(f"x_{index}" for index in range(point1.shape[0])),
        )
        if noise != 0.0:
            x += 2 * noise * (rng.random(x.shape) - np.full(x.shape, 0.5))
        return cls(x, y)

    @staticmethod
    def interpolated_point(point1: ArrayLike, point2: ArrayLike, alpha: float) -> np.ndarray:
        """Creates a point along the line segment between two points.

        Parameters
        ----------
        `point1` : `ArrayLike`
            An array of scalar values.
        `point2` : `ArrayLike`
            An array of scalar values. Should be the same shape as `point1`.
        `alpha` : `float`
            Where to place the new point. If 0.0, new will be the same as `point1`.
            If 1.0, new will be the same as `point2`. If 0.5, new will be at the
            average position of the two points.

        Returns
        -------
        `interpolated_point` : `numpy.ndarray`
            A point on the line segment between `point1` and `point2`.
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        return point1 + alpha * (point2 - point1)

    @staticmethod
    @overload
    def random_points(
        n_points: None = None,
        n_dimensions: int = 2,
        random_state: int | None = None,
    ) -> np.ndarray: ...
    @staticmethod
    @overload
    def random_points(
        n_points: int,
        n_dimensions: int = 2,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, ...]: ...
    @staticmethod
    def random_points(
        n_points: int | None = None,
        n_dimensions: int = 2,
        random_state: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Generate random data points within a range.
        The value of each scalar within each point/vector is in the range `[0,1)`

        Parameters
        ----------
        `n_points` : `int`, optional
            The number of points to generate. If None or not given, a single point
            will be returned; otherwise a tuple of points is returned.
        `n_dimensions` : `int`, default 2
            How many dimensions the vector of each point should have.
        `random_state` : `int`, optional
            The random state to use.

        Returns
        -------
        `random_point(s)` : `numpy.ndarray` or `tuple[numpy.ndarray]`
            The randomly generated point or points.
        """
        rng = np.random.default_rng(random_state)
        if n_points is None:
            return rng.random(n_dimensions)
        return tuple(rng.random(n_dimensions) for _ in range(n_points))

    def to_csv(self, filename: str | Path) -> None:
        """Save this dataset as a `csv` file.

        ...
        """
        pd.concat((self.x, self.y), axis=1).to_csv(filename, index=False)

