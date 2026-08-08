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


from typing import Callable, Protocol, Sequence, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
import re
from numpy.typing import ArrayLike
import pandas as pd


@runtime_checkable
class UnfittedTransformer(Protocol):
    def transform(self, _data: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class FittedTransformer(Protocol):
    def fit_transform(self, _data: ArrayLike) -> ArrayLike: ...


Transformer = UnfittedTransformer | FittedTransformer


@dataclass(slots=True)
class ColumnsFilter:
    columns: str
    transformer: Callable[[ArrayLike], ArrayLike] | Transformer | None = None


def data_from_csv(
    file_path: str | Path,
    col_filters: Sequence[ColumnsFilter],
) -> pd.DataFrame:
    file_path = Path(file_path)
    frames = []
    for col_filter in col_filters:
        frame = pd.read_csv(
            file_path,
            usecols=lambda col: re.search(
                col_filter.columns,
                col,
            ) is not None,
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
