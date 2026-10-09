import sys
import contextlib
from dataclasses import dataclass
from pathlib import Path
import tomllib
import pandas as pd
from liger import dataset as ds
from liger.embedding.sentencetrans import STEmbedder


@dataclass(slots=True)
class Config:
    result_path: Path
    model: str
    texts: pd.Series

    def __init__(self, cfg_path: Path) -> None:
        self.result_path = cfg_path.with_suffix(".csv")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.model = parsed["model"]
        self.texts = ds.data_from_csv(**parsed["content"]).iloc[:, 0]


def main():
    cfg_path = Path(sys.argv[1])
    with contextlib.chdir(cfg_path.parent):
        cfg = Config(Path(cfg_path.name))
        embedder = STEmbedder(cfg.model)
        embeddings = embedder.embed(cfg.texts)
        embeddings.to_csv(cfg.result_path, index=False)


if __name__ == "__main__":
    main()
