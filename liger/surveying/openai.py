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


from typing import overload
from pathlib import Path
from os.path import isfile
from openai import OpenAI
import pandas as pd
from .base import BaseSurveyor


OPENAI_KEYFILE = "openai.key"


class OpenAISurveryor(BaseSurveyor):
    def __init__(self, model_str: str, keyfile: str | Path = OPENAI_KEYFILE) -> None:
        keyfile = Path(keyfile)
        if not isfile(keyfile):
            raise FileNotFoundError(f"\"{keyfile}\" is not a file.")
        self.client = OpenAI(api_key=keyfile.read_text().strip())
        self.model = model_str

    def generate_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
        if response is None:
            raise ValueError("No response was returned")
        return response

    @overload
    def generate_responses(self, prompt: str) -> str: ...
    @overload
    def generate_responses(self, prompt: str, reps: int) -> list[str]: ...
    def generate_responses(self, prompt: str, reps: int | None = None) -> str | list[str]:
        if reps is None:
            return self.generate_response(prompt)
        return [self.generate_response(prompt) for _ in range(reps)]

    def survey(
        self,
        prompts: list[str],
        reps: int = 1,
        allow_dupes: bool = False
    ) -> pd.DataFrame:
        self.check_prompts(prompts, allow_dupes)
        responses: list[list[str]] = []
        for prompt in prompts:
            responses.append(self.generate_responses(prompt, reps))
        return pd.DataFrame({"prompt": prompts, "response": responses})


    def generate_responses(self, prompt: str, reps: int) -> str | list[str]:
        if reps <= 1:
            return self.generate_response(prompt)
        return [self.generate_response(prompt) for _ in range(reps)]


    def generate_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
        if response is None:
            raise ValueError("No response was returned")
        return response

