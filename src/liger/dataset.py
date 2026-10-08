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


from typing import (
    Any,
    Callable,
    Iterable,
    Protocol,
    Sequence,
    TypedDict,
    runtime_checkable
)
import warnings
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
from pandas._typing import UsecolsArgType


@runtime_checkable
class UnfittedTransformer(Protocol):
    def transform(self, data: Any, /) -> Any: ...


@runtime_checkable
class FittedTransformer(Protocol):
    def fit_transform(self, data: Any, /) -> Any: ...


Transformer = Callable[..., Any] | UnfittedTransformer | FittedTransformer


class DataFromCSVArgs(TypedDict):
    columns: str | UsecolsArgType | None
    transformer: Transformer | None


@dataclass(slots=True)
class ColumnsFilter:
    columns: str
    transformer: (
        Callable[[pd.DataFrame], pd.Series | pd.DataFrame] | Transformer | None
    ) = None


def _old_data_from_csv(
    file_path: str | Path,
    col_filters: Sequence[ColumnsFilter],
) -> pd.DataFrame:
    file_path = Path(file_path)
    frames = []
    for col_filter in col_filters:
        frame = pd.read_csv(
            file_path,
            usecols=lambda col: re.search(col_filter.columns, col) is not None,
        )
        if isinstance(col_filter.transformer, UnfittedTransformer):
            frames.append(col_filter.transformer.transform(frame))
        elif isinstance(col_filter.transformer, FittedTransformer):
            frames.append(col_filter.transformer.fit_transform(frame))
        elif isinstance(col_filter.transformer, Callable):
            frames.append(col_filter.transformer(frame))
        else:
            frames.append(frame)
    return pd.concat(frames, axis=1)


def transform_data(
    data: pd.DataFrame,
    transformer: Transformer,
) -> pd.DataFrame:
    if isinstance(transformer, FittedTransformer):
        transformed = transformer.fit_transform(data)
    elif isinstance(transformer, UnfittedTransformer):
        transformed = transformer.transform(data)
    else:
        transformed = transformer(data)
    return pd.DataFrame(transformed)


def extract_columns(
    data: pd.DataFrame,
    columns: str | UsecolsArgType | None = None,
    transformer: Transformer | None = None,
) -> pd.DataFrame:
    if columns is None:
        filtered = pd.DataFrame(data.iloc[:, [0]])
    elif isinstance(columns, str):
        filtered = data.filter(regex=columns, axis=1)
    else:
        filtered = data.filter(items=columns, axis=1)
    if transformer is None:
        return filtered
    return transform_data(filtered, transformer)


def _new_data_from_csv(
    file_path: str | Path,
    columns: str | UsecolsArgType | None = None,
    transformer: Transformer | None = None,
) -> pd.DataFrame:
    file_path = Path(file_path)
    if columns is None:
        data = pd.read_csv(file_path, usecols=[0])
    elif isinstance(columns, str):
        data = pd.read_csv(
            file_path,
            usecols=lambda col: re.search(columns, col) is not None,
        )
    else:
        data = pd.read_csv(file_path, usecols=columns)
    if transformer is None:
        return data
    return transform_data(data, transformer)


def data_from_csv(
    file_path: str | Path,
    col_filters: Sequence[ColumnsFilter] | None = None,
    columns: str | UsecolsArgType | None = None,
    transformer: Transformer | None = None,
) -> pd.DataFrame:
    if col_filters is None:
        return _new_data_from_csv(file_path, columns, transformer)
    if columns is not None or transformer is not None:
        raise ValueError("Received old-typical and new-typical arguments; "
            "Prefer supplying only arguments 'columns' and/or 'transformer'.")
    warnings.warn(
        "Passing 'col_filters' to "
            f"'{data_from_csv.__module__}.{data_from_csv.__name__}' is deprecated; "
            "use arguments 'columns' and/or 'transformer'",
        DeprecationWarning,
    )
    return _old_data_from_csv(file_path, col_filters)


def multi_data_from_csv(
    file_path: str | Path,
    argsets: Iterable[DataFromCSVArgs],
) -> pd.DataFrame:
    return pd.concat((data_from_csv(file_path, **argset) for argset in argsets), axis=1)
