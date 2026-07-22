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


from typing import overload, Any, MutableSequence, Self
from importlib import import_module
import inspect
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


class Dataset:
    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        x_transformers: list[Any] | None = None,
        y_transformers: list[Any] | None = None,
        x_transformers_kwargs: list[dict[str, Any] | None] | None = None,
        y_transformers_kwargs: list[dict[str, Any] | None] | None = None,
    ):
        self.x = x
        self.y = y
        self.x_transformers = self._init_transformers(
            x_transformers,
            x_transformers_kwargs
        )
        self.y_transformers = self._init_transformers(
            y_transformers,
            y_transformers_kwargs,
        )
        self.x = self._transform_data(x, self.x_transformers)
        self.y = self._transform_data(y, self.y_transformers)

    def __repr__(self):
        return f"""Dataset:
x:
{self.x}
y:
{self.y}
x_transformer:
{self.x_transformers}
y_transformer:
{self.y_transformers}"""

    @staticmethod
    def _to_set(input: str | MutableSequence[str] | set[str] | None) -> set[str] | None:
        if input is None:
            return None
        if isinstance(input, str):
            return {input,}
        if isinstance(input, MutableSequence):
            return {string for string in input}
        return input

    @staticmethod
    def _patterns_in(string: str, patterns: set[str] | None) -> bool:
        if patterns is None:
            return False
        return any(pattern in string for pattern in patterns)

    @staticmethod
    def _filter_cols(cols: pd.Index, patterns: set[str] | None) -> pd.Index:
        if patterns is None:
            return pd.Index([])
        return pd.Index(col for col in cols if Dataset._patterns_in(col, patterns))

    @staticmethod
    def _init_transformer(transformer: Any, kwargs: dict[str, Any] | None) -> Any:
        if isinstance(transformer, str):
            split_transformer = transformer.rsplit(".", 1)
            transformer = getattr(
                import_module(split_transformer[0]),
                split_transformer[1],
            )
        if kwargs is None:
            kwargs = {}
        if isinstance(transformer, type):
            return transformer(**kwargs).set_output(transform="pandas")
        elif inspect.isfunction(transformer):
            return FunctionTransformer(
                transformer,
                kw_args=kwargs,
            ).set_output(transform="pandas")
        return transformer

    @classmethod
    def _init_transformers(
        cls,
        transformers: list[Any] | None,
        kwargs: list[dict[str, Any] | None] | None,
    ) -> list[Any] | None:
        if transformers is None or len(transformers) == 0:
            return None
        if kwargs is None or len(kwargs) == 0:
            kwargs = [{} for _ in range(len(transformers))]
        elif len(kwargs) != len(transformers):
            raise ValueError(
                f"length of transformers is {len(transformers)}, "
                f"but length of transformers kwargs is {len(kwargs)}"
            )
        return [
            cls._init_transformer(transformer, kwargs)
            for transformer, kwargs in zip(transformers, kwargs)
        ]

    @staticmethod
    def _transform_data(
            data: pd.DataFrame,
            transformers: list[Any] | None,
    ) -> pd.DataFrame:
        if transformers is None or len(transformers) == 0 or all(
            tf is None
            for tf in transformers
        ):
            return data
        for transformer in transformers:
            transformer.fit(data)
        return pd.concat([
            pd.DataFrame(transformer.transform(data), copy=False)
            for transformer in transformers if transformer is not None
        ], axis=1)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        x_patterns: str | MutableSequence[str] | set[str] | None = None,
        y_patterns: str | MutableSequence[str] | set[str] | None = None,
        x_transformers: list[Any] | None = None,
        y_transformers: list[Any] | None = None,
        x_transformers_kwargs: list[dict[str, Any] | None] | None = None,
        y_transformers_kwargs: list[dict[str, Any] | None] | None = None,
    ) -> Self:
        """
        Initialize a `Dataset` from a `pandas.DataFrame`.
        #
        Parameters
        ----------
        `df` : `pandas.DataFrame`
            The DataFrame with the desired data.
        `x_patterns` : `str | MutableSequence[str] | set[str]`, optional
            The pattern(s) to search for in column names to use for x data.
        `y_patterns` : `str | MutableSequence[str] | set[str]`, optional
            The pattern(s) to search for in column names to use for y data.
        `x_transformers` : `list[Any]`, optional
            Function(s) to apply to the x data.
            Outputs from multiple functions will be concatenated column-wise.
        `y_transformers` : `list[Any]`, optional
            Function(s) to apply to the y data.
            Outputs from multiple functions will be concatenated column-wise.
        `x_transformers_kwargs` : `list[dict[str, Any | None]`, optional
            Keyword arguments to pass to x transformers.
        `y_transformers_kwargs` : `list[dict[str, Any | None]`, optional
            Keyword arguments to pass to y transformers.
        #
        Returns
        -------
        `dataset` : `Dataset`
            A dataset with data matching the csv file contents,
            potentially filtered (see above).
        """
        x_patterns = cls._to_set(x_patterns)
        y_patterns = cls._to_set(y_patterns)
        x = df.filter(cls._filter_cols(df.columns, x_patterns))
        if x_patterns is not None and x.empty:
            raise ValueError("No data in x")
        y = df.filter(cls._filter_cols(df.columns, y_patterns))
        if y_patterns is not None and y.empty:
            raise ValueError("No data in y")
        return cls(
            x,
            y,
            x_transformers,
            y_transformers,
            x_transformers_kwargs,
            y_transformers_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        file_path: str | Path,
        x_patterns: str | MutableSequence[str] | set[str] | None = None,
        y_patterns: str | MutableSequence[str] | set[str] | None = None,
        x_transformers: list[Any] | None = None,
        y_transformers: list[Any] | None = None,
        x_transformers_kwargs: list[dict[str, Any] | None] | None = None,
        y_transformers_kwargs: list[dict[str, Any] | None] | None = None,
    ) -> Self:
        """
        Initialize a `Dataset` from a `csv` file.
        #
        Parameters
        ----------
        `file_path` : `str`
            The path of the `csv` file to read.
        `x_patterns` : `str | MutableSequence[str] | set[str]`, optional
            The pattern(s) to search for in column names to use for x data.
            NOTE: If both `x_patterns` and `y_patterns` are provided,
            only columns matching them will be loaded into memory.
        `y_patterns` : `str | MutableSequence[str] | set[str]`, optional
            The pattern(s) to search for in column names to use for y data.
            NOTE: If both `x_patterns` and `y_patterns` are provided,
            only columns matching them will be loaded into memory.
        `x_transformers` : `list[Any]`, optional
            Function(s) to apply to the x data.
            Outputs from multiple functions will be concatenated column-wise.
        `y_transformers` : `list[Any]`, optional
            Function(s) to apply to the y data.
            Outputs from multiple functions will be concatenated column-wise.
        `x_transformers_kwargs` : `list[dict[str, Any | None]`, optional
            Keyword arguments to pass to x transformers.
        `y_transformers_kwargs` : `list[dict[str, Any | None]`, optional
            Keyword arguments to pass to y transformers.
        #
        Returns
        -------
        `dataset` : `Dataset`
            A dataset with data matching the csv file contents,
            potentially filtered (see above).
        """
        file_path = Path(file_path)
        x_patterns = cls._to_set(x_patterns)
        y_patterns = cls._to_set(y_patterns)
        x = pd.read_csv(
            file_path,
            usecols=lambda col: cls._patterns_in(col, x_patterns),
        )
        if x_patterns is not None and x.empty:
            raise ValueError("No data in x")
        y = pd.read_csv(
            file_path,
            usecols=lambda col: cls._patterns_in(col, y_patterns),
        )
        if y_patterns is not None and y.empty:
            raise ValueError("No data in y")
        return cls(
            x,
            y,
            x_transformers,
            y_transformers,
            x_transformers_kwargs,
            y_transformers_kwargs,
        )

    @staticmethod
    def interpolated_point(
            point1: ArrayLike,
            point2: ArrayLike,
            alpha: float,
    ) -> np.ndarray:
        """Creates a point along the line segment between two points.
        #
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
        #
        Returns
        -------
        `interpolated_point` : `numpy.ndarray`
            A point on the line segment between `point1` and `point2`.
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        return point1 + alpha * (point2 - point1)

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
        points, with targets corresponding to the distance of each point along that
        line segment and noise applied to each point.
        #
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
            np.stack([cls.interpolated_point(
                point1,
                point2,
                alpha,
            ) for alpha in y.iloc[:, 0]]),
            columns=pd.Index(f"x_{index}" for index in range(point1.shape[0])),
        )
        if noise != 0.0:
            x += 2 * noise * (rng.random(x.shape) - np.full(x.shape, 0.5))
        return cls(x, y)

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
        #
        Parameters
        ----------
        `n_points` : `int`, optional
            The number of points to generate. If None or not given, a single point
            will be returned; otherwise a tuple of points is returned.
        `n_dimensions` : `int`, default 2
            How many dimensions the vector of each point should have.
        `random_state` : `int`, optional
            The random state to use.
        #
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
        #
        ...
        """
        pd.concat((self.x, self.y), axis=1).to_csv(filename, index=False)
