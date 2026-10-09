from typing import Any, TypedDict
import sys
import contextlib
from dataclasses import dataclass
from pathlib import Path
import tomllib
import itertools
import pandas as pd
from liger import dataset as ds


class CSVColumn(TypedDict):
    path: str
    column: str | None


@dataclass(slots=True)
class Config:
    form: str
    fields_array: list[list[dict[str, Any]]]
    result_path: Path
    result_column: str

    def __init__(self, cfg_path: Path) -> None:
        self.result_path = cfg_path.with_suffix(".csv")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.form = parsed["form"]
        self.result_column = parsed["result_column"]
        self.fields_array = [
            self._init_fields_axis(fields_axis)
            for fields_axis in parsed["fields_array"]
        ]

    @staticmethod
    def _init_fields_axis(
        fields: dict[str, list[str] | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expanded_fields_axis = {
            key: (
                values
                if isinstance(values, list)
                else ds.data_from_csv(**values).iloc[:, 0]
            )
            for key, values in fields.items()
        }
        return [{
            key: value
            for key, value in zip(expanded_fields_axis.keys(), values)
        } for values in zip(*expanded_fields_axis.values())]


def _concat_kwargs(*args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for arg in args for key, value in arg.items()}


def main():
    cfg_path = Path(sys.argv[1])
    with contextlib.chdir(cfg_path.parent):
        cfg = Config(Path(cfg_path.name))
        prompts = pd.Series([
            cfg.form.format(**_concat_kwargs(*fields))
            for fields in itertools.product(*cfg.fields_array)
        ], name=cfg.result_column)
        prompts.to_csv(cfg.result_path, index=False)


if __name__ == "__main__":
    main()
