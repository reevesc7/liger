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


from typing import Any
from collections.abc import Callable
from importlib import import_module
from functools import singledispatch
from pathlib import Path
import numpy as np
import pandas as pd


def import_from_str(import_path: str) -> Any:
    try:
        return import_module(import_path)
    except ImportError:
        pass
    path_parts = import_path.split(".")
    for part_index in range(len(path_parts) - 1, 0, -1):
        module_path = ".".join(path_parts[:part_index])
        try:
            obj = import_module(module_path)
        except ImportError:
            continue
        for attr_name in path_parts[part_index:]:
            try:
                obj = getattr(obj, attr_name)
            except AttributeError:
                raise ImportError(f"{obj!r} has no attribute {attr_name!r}")
        return obj
    raise ImportError(f"Could not import {import_path!r}")


@singledispatch
def to_json_compatible(obj: Any) -> Any:
    raise TypeError(
        f"No JSON encoder registered for type {type(obj).__name__!r}. "
        "A relevant encoder may be registered in a 'liger.serde' submodule."
    )

@to_json_compatible.register(range)
def _(obj: range) -> list:
    return list(obj)


@to_json_compatible.register(Callable)
def _(obj: Callable) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    name = getattr(obj, "__name__", None)
    if not(module is None or qualname is None):
        return f"{module}.{qualname}"
    elif not(module is None or name is None):
        return f"{module}.{name}"
    elif name is not None:
        return name
    else:
        return repr(obj)


@to_json_compatible.register(Path)
def _(obj: Path) -> str:
    return str(obj)


@to_json_compatible.register(np.ndarray)
def _(obj: np.ndarray) -> list:
    return obj.tolist()


@to_json_compatible.register(np.generic)
def _(obj: np.generic) -> Any:
    return obj.item()


@to_json_compatible.register(pd.Series)
def _(obj: pd.Series) -> dict:
    return obj.to_dict()


@to_json_compatible.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> dict:
    return obj.to_dict()
