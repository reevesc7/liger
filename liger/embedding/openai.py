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


from typing import MutableSequence, overload
from pathlib import Path
import pandas as pd
from openai import OpenAI
from .base import BaseEmbedder


OPENAI_KEYFILE = "openai.key"


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_str: str, keyfile: str | Path = OPENAI_KEYFILE):
        keyfile = Path(keyfile)
        if not keyfile.is_file():
            raise FileNotFoundError(f"No \"{OPENAI_KEYFILE}\" in the working directory")
        self.client = OpenAI(api_key=keyfile.read_text().strip())
        self.model = model_str

    def _embed_one(self, string: str) -> list[float]:
        print(f"OpenAIEmbedder embedding \"{string[:16]}...{string[-16:]}\"".replace("\n", " "))
        return self.client.embeddings.create(
            input=string.replace("\n", " "),
            model=self.model,
        ).data[0].embedding

    def _col_names(self, n_dims: int) -> pd.Index:
        return pd.Index(f"{self.model}_{dim}" for dim in range(n_dims))

    @overload
    def embed(self, strings: str) -> pd.Series: ...
    @overload
    def embed(self, strings: MutableSequence[str] | pd.Series) -> pd.DataFrame: ...
    def embed(self, strings: str | MutableSequence[str] | pd.Series) -> pd.Series | pd.DataFrame:
        if isinstance(strings, str):
            embedding = pd.Series(self._embed_one(strings))
            embedding.index = self._col_names(embedding.size)
            return embedding
        embeddings = pd.DataFrame((self._embed_one(string) for string in strings))
        embeddings.columns = self._col_names(embeddings.shape[1])
        return embeddings

