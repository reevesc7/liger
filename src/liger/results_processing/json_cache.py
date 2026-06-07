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


from typing import Any, Iterable, Iterator
from pathlib import Path
import json


class JSONCache:
    def __init__(
        self,
        paths: Path | str | Iterable[Path | str],
        patterns: str | Iterable[str],
    ) -> None:
        if isinstance(paths, (Path, str)):
            paths = paths,
        if isinstance(patterns, str):
            patterns = patterns,
        filepaths: set[Path] = set()
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{path} is not a file or directory")
            filepaths.update(self._filepaths_matching_patterns(path, patterns))
        self.filepaths = sorted(filepaths)
        self._cache: dict[Path, Any] = {}

    @staticmethod
    def _filepaths_matching_patterns(path: Path, patterns: Iterable[str]) -> set[Path]:
        if path.is_file() and any(path.match(pattern) for pattern in patterns):
            return {path}
        return {
            fpath
            for pattern in patterns
            for fpath in sorted(path.rglob(pattern))
            if fpath.is_file()
        }

    def get_file_data(self, filepath: Path | str) -> Any:
        filepath = Path(filepath)
        if filepath not in self._cache:
            if filepath not in self.filepaths:
                self.filepaths.append(filepath)
            with open(filepath, "r") as file:
                self._cache[filepath] = json.load(file)
        return self._cache[filepath]

    def __iter__(self) -> Iterator[tuple[Path, Any]]:
        for filepath in self.filepaths:
            yield filepath, self.get_file_data(filepath)

