from typing import Any, Sequence, overload
from pathlib import Path
import os
import json


def _dirpaths_and_filepaths_to_filepaths(paths: Sequence[Path]) -> set[Path]:
    filepaths: set[Path] = set()
    for path in paths:
        if not path.exists():
            print(f"Warning: {path} is not an existing directory or file")
        elif path.is_file():
            filepaths.add(path)
        elif not path.is_dir():
            print(f"Warning: {path} is exists but is not a directory or file")
            continue
        for base, _, names in os.walk(path):
            filepaths.update({Path(base, name) for name in names})
    return filepaths


def mass_json_load(
    paths: Path | str | Sequence[Path | str],
    read_all_json_files: bool = False,
    filenames_to_read: str | set[str] = "pipeline_data.json",
) -> list[Any]:
    if isinstance(paths, Path):
        paths_f = [paths]
    elif isinstance(paths, str):
        paths_f = [Path(paths)]
    elif isinstance(paths, Sequence):
        paths_f = [Path(path) for path in paths]
    else:
        raise TypeError("`paths` must be str, Path, or sequence of str or Path")
    if isinstance(filenames_to_read, str):
        filenames_to_read = {filenames_to_read,}
    filepaths = _dirpaths_and_filepaths_to_filepaths(paths_f)
    outputs: list[Any] = []
    for filepath in filepaths:
        if (
            not (read_all_json_files or filepath.name in filenames_to_read)
            or filepath.name.rsplit(".", 1)[-1] != "json"
        ):
            continue
        with open(filepath, "r") as file:
            outputs.append(json.load(file))
    return outputs


_sentinel = object()

@overload
def mass_dict_get(dicts: Sequence[dict[str, Any]], key: str) -> list[Any]: ...
@overload
def mass_dict_get(dicts: Sequence[dict[str, Any]], key: str, default: Any) -> list[Any]: ...

def mass_dict_get(dicts: Sequence[dict[str, Any]], key: str, default: Any = _sentinel) -> list[Any]:
    if default is _sentinel:
        return [dictionary.get(key) for dictionary in dicts]
    return [dictionary.get(key, default) for dictionary in dicts]

