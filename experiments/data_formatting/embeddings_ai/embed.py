import sys
from dataclasses import dataclass
from pathlib import Path
import tomllib
import pandas as pd
from liger.embedding.openai import OpenAIEmbedder


@dataclass(slots=True)
class Config:
    cfg_dir: Path
    result_path: Path
    model: str
    texts: pd.Series

    def __init__(self, cfg_path: Path) -> None:
        self.cfg_dir = cfg_path.parent
        self.result_path = cfg_path.with_suffix(".csv")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.model = parsed["model"]
        self.texts = self._read_column(**parsed["content"])

    def _read_column(self, path: str, column: str | None = None) -> pd.Series:
        if column is None:
            return pd.Series(pd.read_csv(self.cfg_dir / path).iloc[:, 0])
        return pd.Series(pd.read_csv(
            self.cfg_dir / path,
            usecols=lambda col: col == column,
        )[column])


def main():
    cfg = Config(Path(sys.argv[1]))
    embedder = OpenAIEmbedder(cfg.model)
    embeddings = embedder.embed(cfg.texts)
    embeddings.to_csv(cfg.result_path)


if __name__ == "__main__":
    main()
