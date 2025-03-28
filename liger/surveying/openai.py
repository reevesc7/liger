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


from pathlib import Path
from os.path import isfile
from openai import OpenAI
import pandas as pd
from .base import BaseSurveyor


OPENAI_KEYFILE = "openai.key"


class OpenAISurveryor(BaseSurveyor):
    def __init__(self, model_str: str) -> None:
        if not isfile(Path(OPENAI_KEYFILE)):
            raise FileNotFoundError(
                f"No \"{OPENAI_KEYFILE}\" in the working directory"
            )
        self.client = OpenAI(api_key=Path(OPENAI_KEYFILE).read_text().strip())
        self.model = model_str


    def survey(
        self,
        prompts: list[str],
        reps: int = 1,
        allow_dupes: bool = False
    ) -> pd.DataFrame:
        OpenAISurveryor.check_prompts(prompts, allow_dupes)
        responses = []
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

