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


from typing import Any, Callable, Final, Mapping, Sequence, TypeVar, overload
from importlib import import_module
from functools import partial
from liger.typing import LgConfigLike, is_lg_config_like
from ._constants import OBJECT, INSTANCE, PARTIAL, ARGS


T = TypeVar("T")


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET: Final = _UnsetType()


def _import(import_path: str) -> Any:
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


def _get_args(config: Mapping[str, LgConfigLike]) -> tuple[Any, ...]:
    if ARGS not in config:
        return ()
    args = config[ARGS]
    if not isinstance(args, Sequence) or isinstance(args, str):
        raise TypeError(
            f"{ARGS!r} key must map to {Sequence.__name__!r}, "
                f"but is {type(args).__name__!r}"
        )
    return tuple(parse_config(arg) for arg in args)


def _get_kwargs(config: Mapping[str, LgConfigLike]) -> dict[str, Any]:
    return {
        key: parse_config(value)
        for key, value in config.items()
        if key not in (INSTANCE, ARGS, PARTIAL)
    }


def _instantiate(config: Mapping[str, LgConfigLike]) -> Any:
    factory: Callable = _import(str(config[INSTANCE]))
    return factory(*_get_args(config), **_get_kwargs(config))


def _partialize(config: Mapping[str, LgConfigLike]) -> partial:
    func: Callable = _import(str(config[PARTIAL]))
    return partial(func, *_get_args(config), **_get_kwargs(config))


@overload
def parse_config(config: LgConfigLike) -> Any: ...
@overload
def parse_config(config: LgConfigLike, expected_type: type[T]) -> T: ...
def parse_config(
        config: LgConfigLike,
        expected_type: type[T] | _UnsetType = UNSET,
) -> Any:
    if not is_lg_config_like(config):
        raise TypeError(f"'config' must be {LgConfigLike!r}")
    if isinstance(config, Sequence) and not isinstance(config, str):
        obj = [parse_config(element) for element in config]
    elif isinstance(config, Mapping):
        if OBJECT in config:
            obj = _import(str(config[OBJECT]))
        elif INSTANCE in config:
            obj = _instantiate(config)
        elif PARTIAL in config:
            obj = _partialize(config)
        else:
            obj = {key: parse_config(value) for key, value in config.items()}
    else:
        obj = config
    if not (isinstance(expected_type, _UnsetType) or isinstance(obj, expected_type)):
        raise TypeError(
            f"Expected type {expected_type.__name__!r}, "
                f"but parsed config is type {type(obj).__name__!r}"
        )
    return obj
