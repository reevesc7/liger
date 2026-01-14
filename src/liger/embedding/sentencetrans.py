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
import pandas as pd
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class STEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_str: str,
    ):
        self.model_str = model_str
        self.model = SentenceTransformer(model_str)

    @overload
    def embed(self, strings: str) -> pd.Series: ...
    @overload
    def embed(self, strings: MutableSequence[str] | pd.Series) -> pd.DataFrame: ...
    def embed(self, strings: str | MutableSequence[str] | pd.Series) -> pd.Series | pd.DataFrame:
        print(f"STEmbedder embedding...")
        model_dims = self.model.get_sentence_embedding_dimension()
        if model_dims is None:
            raise ValueError("Model did not return number of embedding dimensions")
        cols = pd.Index(f"{self.model_str}_{dim}" for dim in range(model_dims))
        if isinstance(strings, (MutableSequence, pd.Series)):
            strings = list(strings)
        elif isinstance(strings, str):
            return pd.Series(self.model.encode(strings, convert_to_numpy=True), index=cols)
        return pd.DataFrame(self.model.encode(strings), columns=cols)

