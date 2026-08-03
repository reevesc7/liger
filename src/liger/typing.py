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


from typing import Any, Mapping, Sequence, TypeGuard


type LgConfig = (
    bool
        | int
        | float
        | str
        | list[LgConfig]
        | dict[str, LgConfig]
        | None
)


type LgConfigLike = (
    bool
        | int
        | float
        | str
        | Sequence[LgConfigLike]
        | Mapping[str, LgConfigLike]
        | None
)


def is_lg_config(obj: Any) -> TypeGuard[LgConfig]:
    if obj is None or isinstance(obj, bool | int | float | str):
        return True
    if isinstance(obj, list):
        return all(is_lg_config(element) for element in obj)
    if isinstance(obj, dict):
        return (
            all(isinstance(key, str) for key in obj.keys())
            and
            all(is_lg_config(value) for value in obj.values())
        )
    return False


def is_lg_config_like(obj: Any) -> TypeGuard[LgConfigLike]:
    if obj is None or isinstance(obj, bool | int | float | str):
        return True
    if isinstance(obj, Sequence):
        return all(is_lg_config(element) for element in obj)
    if isinstance(obj, Mapping):
        return (
            all(isinstance(key, str) for key in obj.keys())
            and
            all(is_lg_config(value) for value in obj.values())
        )
    return False
