from typing import TypedDict
import sys
from dataclasses import dataclass
from pathlib import Path
import tomllib
import re
import pandas as pd


class CSVColumn(TypedDict):
    path: str
    column: str | None


@dataclass(slots=True)
class Config:
    cfg_dir: Path
    result_path: Path
    data: pd.DataFrame

    def __init__(self, cfg_path: Path) -> None:
        self.cfg_dir = cfg_path.parent
        self.result_path = cfg_path.with_suffix(".csv")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.data = pd.concat(
            (self._read_columns(**col) for col in parsed["columns"]),
            axis=1,
        )

    def _read_columns(self, path: str, column: str | None = None) -> pd.DataFrame:
        if column is None:
            return pd.read_csv(self.cfg_dir / path).iloc[:, 0]
        return pd.read_csv(
            self.cfg_dir / path,
            usecols=lambda col: re.search(column, col) is not None,
        )


def main():
    cfg = Config(Path(sys.argv[1]))
    cfg.data.to_csv(cfg.result_path, index=False)


if __name__ == "__main__":
    main()
