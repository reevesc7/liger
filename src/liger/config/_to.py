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


from typing import Any, TypeVar
from collections.abc import Callable, Iterable, Mapping
import inspect
from functools import partial, singledispatch
from pathlib import Path
import numpy as np
import pandas as pd
from liger.typing import LgConfig
from ._constants import OBJECT, INSTANCE, PARTIAL, ARGS


VT = TypeVar("VT", bool, int, float, str, None)


def _attr_path(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


@singledispatch
def to_config(obj: Any) -> LgConfig:
    raise TypeError(f"No {LgConfig.__name__!r} encoder registered for "
        f"type {type(obj).__name__!r}. Type can be registered with "
        f"'@{to_config.__module__}.{to_config.__qualname__}"
        f".register({type(obj).__name__})'")


def instance_to_config(factory: Callable, *args, **kwargs) -> LgConfig:
    config: dict[str, Any] = {INSTANCE: _attr_path(factory)}
    if args:
        config[ARGS] = args
    return to_config(config | kwargs)


def instance_init_args_to_config(obj: Any) -> LgConfig:
    signature = inspect.signature(type(obj))
    return instance_to_config(type(obj), **{
        key: getattr(obj, key)
        for key in signature.parameters.keys() if key != "kwargs"
    })


@to_config.register(bool | int | float | str | None)
def _(obj: VT) -> VT:
    return obj


@to_config.register(Iterable)
def _(obj: Iterable) -> list[LgConfig]:
    return [to_config(element) for element in obj]


@to_config.register(Mapping)
def _(obj: Mapping) -> dict[str, LgConfig]:
    return {str(key): to_config(value) for key, value in obj.items()}


@to_config.register
def _(obj: Callable) -> dict[str, LgConfig]:
    if not hasattr(obj, "__qualname__"):
        raise TypeError(f"No {LgConfig!r} encoder registered for callable instance of"
            f"type {type(obj).__name__!r}. Type can be registered with "
            f"'@{to_config.__module__}.{to_config.__qualname__}"
            f".register({type(obj).__name__})'")
    return {OBJECT: _attr_path(obj)}
    # if hasattr(obj, "__qualname__"):
    #     name = obj.__qualname__
    # elif hasattr(obj, "__name__"):
    #     name = obj.__name__
    # else:
    #     raise TypeError(
    #         f"No {Config!r} encoder registered for "
    #             f"callable type {type(obj).__name__!r}. "
    #             f"A relevant encoder may be registered in a {__name__!r} submodule."
    #     )
    # return {OBJECT_TAG: f"{obj.__module__}.{name}"}


@to_config.register(partial)
def _(obj: partial) -> dict[str, LgConfig]:
    func: dict[str, str] = {PARTIAL: _attr_path(obj.func)}
    attrs: dict[str, Any] = {ARGS: obj.args} if obj.args else {}
    config = to_config(attrs | obj.keywords)
    if not isinstance(config, dict):
        raise TypeError(
            f"Config should by type 'dict', but is type '{type(config).__name__!r}"
        )
    return func | config


@to_config.register(range)
def _(obj: range) -> LgConfig:
    return instance_to_config(range, obj.start, obj.stop, obj.step)


@to_config.register(Path)
def _(obj: Path) -> str:
    return str(obj.resolve())


@to_config.register(np.generic)
def _(obj: np.generic) -> Any:
    return obj.item()


@to_config.register(pd.RangeIndex)
def _(obj: pd.RangeIndex) -> LgConfig:
    return instance_to_config(type(obj), start=obj.start, stop=obj.stop, step=obj.step)


@to_config.register(pd.Index)
def _(obj: pd.Index) -> LgConfig:
    return instance_to_config(type(obj), data=obj.to_list(), name=obj.name)


@to_config.register(pd.Series)
def _(obj: pd.Series) -> LgConfig:
    return instance_to_config(
        type(obj),
        data=obj.to_list(),
        index=obj.index,
        name=obj.name,
    )


@to_config.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> LgConfig:
    return instance_to_config(
        type(obj),
        data=obj.to_numpy(),
        index=obj.index,
        columns=obj.columns,
    )
