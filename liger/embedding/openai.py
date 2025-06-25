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


from typing import MutableSequence
from pathlib import Path
from openai import OpenAI
import numpy as np
from .base import BaseEmbedder


OPENAI_KEYFILE = "openai.key"


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_str: str, keyfile: str | Path = OPENAI_KEYFILE):
        keyfile = Path(keyfile)
        if not keyfile.is_file():
            raise FileNotFoundError(f"No \"{OPENAI_KEYFILE}\" in the working directory")
        self.client = OpenAI(api_key=keyfile.read_text().strip())
        self.model = model_str

    def embed(self, strings: str | MutableSequence[str] | np.ndarray) -> np.ndarray:
        if isinstance(strings, str):
            print(f"OpenAIEmbedder embedding \"{strings[:16]}\"...".replace("\n", " "))
            return np.array(self.client.embeddings.create(
                input=strings.replace("\n", " "),
                model=self.model
            ).data[0].embedding)
        return np.array([self.embed(string) for string in strings])

