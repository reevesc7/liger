from typing import Any, NamedTuple, TypedDict
import sys
from pathlib import Path
import tomllib
import itertools
import pandas as pd


DIR = Path(__file__).parent.resolve()


class CSVColumn(TypedDict):
    path: str
    column: str | None


class Config(NamedTuple):
    form: str
    fields_array: list[dict[str, list[str] | CSVColumn]]
    result_path: str
    column: str


def _read_column(path: str, column: str | None = None) -> pd.Series:
    if column is None:
        return pd.Series(
            pd.read_csv(DIR / path).iloc[:, 0]
        )
    return pd.Series(pd.read_csv(
        DIR / path,
        usecols=lambda col: col == column,
    )[column])


def _init_fields_axis(
    fields: dict[str, list[str] | CSVColumn],
) -> list[dict[str, Any]]:
    expanded_fields_axis = {
        key: values if isinstance(values, list) else _read_column(**values)
        for key, values in fields.items()
    }
    return [{
        key: value
        for key, value in zip(expanded_fields_axis.keys(), values)
    } for values in zip(*expanded_fields_axis.values())]


def _concat_kwargs(*args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for arg in args for key, value in arg.items()}


def main():
    cfg = Config(**tomllib.load(Path(sys.argv[1]).open("rb")))
    expanded_fields_array = [
        _init_fields_axis(fields_axis)
        for fields_axis in cfg.fields_array
    ]
    prompts = pd.Series((
        cfg.form.format(**_concat_kwargs(*fields))
        for fields in itertools.product(*expanded_fields_array)
    ), name=cfg.column)
    prompts.to_csv(DIR / cfg.result_path, index=False)


if __name__ == "__main__":
    main()
