import sys
import contextlib
from dataclasses import dataclass
from pathlib import Path
import tomllib
import pandas as pd
from liger import dataset as ds


@dataclass(slots=True)
class Config:
    cfg_dir: Path
    result_path: Path
    data: pd.DataFrame

    def __init__(self, cfg_path: Path) -> None:
        self.result_path = cfg_path.with_suffix(".csv")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.data = pd.concat(
            (ds.data_from_csv(**columns) for columns in parsed["data"]),
            axis=1,
        )


def main():
    cfg_path = Path(sys.argv[1])
    with contextlib.chdir(cfg_path.parent):
        cfg = Config(Path(cfg_path.name))
        cfg.data.to_csv(cfg.result_path, index=False)


if __name__ == "__main__":
    main()
