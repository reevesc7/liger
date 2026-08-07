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
from dataclasses import dataclass, field
from importlib import import_module
from functools import partial
from simpleeval import simple_eval
from liger.typing import LgConfigLike, is_lg_config_like, RawList, RawDict
from ._constants import ConfigTag


T = TypeVar("T")


class UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET: Final = UnsetType()


@dataclass(slots=True)
class _ConfigParser:
    config: LgConfigLike
    kwargs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _detect_multi_key(config: Mapping[str, LgConfigLike], tag: str) -> None:
        if len(config) > 1:
            raise ValueError(f"Found multiple keys in {tag!r}-tagged mapping")

    def _make_raw(self, config: Mapping[str, LgConfigLike]) -> Any:
        self._detect_multi_key(config, ConfigTag.RAW)
        value = config[ConfigTag.RAW]
        if isinstance(value, Sequence) and not isinstance(value, str):
            return RawList(value)
        if isinstance(value, Mapping):
            return RawDict(value)
        return value

    @staticmethod
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

    def _make_object(self, config: Mapping[str, LgConfigLike]) -> Any:
        self._detect_multi_key(config, ConfigTag.OBJECT)
        return self._import(str(config[ConfigTag.OBJECT]))

    def _get_args(self, config: Mapping[str, LgConfigLike]) -> tuple[Any, ...]:
        if ConfigTag.ARGS not in config:
            return ()
        args = config[ConfigTag.ARGS]
        if not isinstance(args, Sequence) or isinstance(args, str):
            raise TypeError(
                f"{ConfigTag.ARGS!r} key must map to {Sequence.__name__!r}, "
                    f"but is {type(args).__name__!r}"
            )
        return tuple(self._parse_config(arg) for arg in args)

    def _get_kwargs(self, config: Mapping[str, LgConfigLike]) -> dict[str, Any]:
        if any(tag.value in config for tag in ConfigTag):
            raise ValueError("Config tags found in keyword arguments "
                f"for an instance or partial:\n{config}")
        return {key: self._parse_config(value) for key, value in config.items()}

    def _make_instance(self, config: Mapping[str, LgConfigLike]) -> Any:
        factory: Callable = self._import(str(config[ConfigTag.INSTANCE]))
        return factory(*self._get_args(config), **self._get_kwargs({
            key: value
            for key, value in config.items()
            if key not in (ConfigTag.ARGS, ConfigTag.INSTANCE)
        }))

    def _make_partial(self, config: Mapping[str, LgConfigLike]) -> partial:
        func: Callable = self._import(str(config[ConfigTag.PARTIAL]))
        return partial(func, *self._get_args(config), **self._get_kwargs({
            key: value
            for key, value in config.items()
            if key not in (ConfigTag.ARGS, ConfigTag.PARTIAL)
        }))

    def _eval(self, config: Mapping[str, LgConfigLike]) -> Any:
        self._detect_multi_key(config, ConfigTag.EVAL)
        return simple_eval(config[ConfigTag.EVAL], names=self.kwargs)

    def _parse_config(self, config: LgConfigLike) -> Any:
        if not is_lg_config_like(config):
            raise TypeError(f"'config' must be {LgConfigLike!r}")
        if isinstance(config, Sequence) and not isinstance(config, str):
            obj = [self._parse_config(element) for element in config]
        elif isinstance(config, Mapping):
            if ConfigTag.RAW in config:
                obj = self._make_raw(config)
            elif ConfigTag.OBJECT in config:
                obj = self._make_object(config)
            elif ConfigTag.INSTANCE in config:
                obj = self._make_instance(config)
            elif ConfigTag.PARTIAL in config:
                obj = self._make_partial(config)
            elif ConfigTag.EVAL in config:
                obj = self._eval(config)
            else:
                obj = {key: self._parse_config(value) for key, value in config.items()}
        else:
            obj = config
        return obj

    @overload
    def parse(self, expected_type: UnsetType = UNSET) -> Any: ...
    @overload
    def parse(self, expected_type: type[T]) -> T: ...
    def parse(self, expected_type: type[T] | UnsetType = UNSET) -> Any:
        obj = self._parse_config(self.config)
        if not (isinstance(expected_type, UnsetType) or isinstance(obj, expected_type)):
            raise TypeError(
                f"Expected type {expected_type.__name__!r}, "
                    f"but parsed config is type {type(obj).__name__!r}"
            )
        return obj


@overload
def parse_config(config: LgConfigLike, **kwargs: Any) -> Any: ...
@overload
def parse_config(config: LgConfigLike, expected_type: type[T], **kwargs: Any) -> T: ...
def parse_config(
    config: LgConfigLike,
    expected_type: type[T] | UnsetType = UNSET,
    **kwargs: Any,
) -> Any:
    return _ConfigParser(config, kwargs).parse(expected_type)
