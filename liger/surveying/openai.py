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
import tiktoken
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
        self.logit_bias: dict[str, int] | None = None

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

    def set_logit_bias(self, desired_tokens: str | set[str]) -> None:
        encoding = tiktoken.encoding_for_model(self.model)
        if isinstance(desired_tokens, str):
            desired_tokens = set(desired_tokens,)
        self.logit_bias = {}
        for token in desired_tokens:
            token_ids = set(encoding.encode(token))
            for token_id in token_ids:
                self.logit_bias[str(token_id)] = 100

    def logprobs(
        self,
        prompt: str,
        response_seed: str,
        allowed_tokens: str | set[str] | None = None,
        normalize: bool = True,
    ) -> list[tuple[str, float]]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_seed},
            ],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            logit_bias=self.logit_bias,
        )
        logprobs = completion.choices[0].logprobs
        if logprobs is not None and logprobs.content is not None:
            response = logprobs.content[0].top_logprobs
        else:
            raise ValueError("No response was returned or response was unparseable")
        if isinstance(allowed_tokens, str):
            allowed_tokens = set(allowed_tokens,)
        if isinstance(allowed_tokens, set):
            response = [logprob for logprob in response if logprob.token in allowed_tokens]
        probs = [10 ** logprob.logprob for logprob in response]
        if normalize:
            total_prob = sum(probs)
            probs = [prob / total_prob for prob in probs]
        return [(logprob.token, prob) for logprob, prob in zip(response, probs)]

    def logprobs_survey(
        self,
        prompts: list[str],
        response_seeds: str | list[str],
        allowed_tokens: str | set[str] | None = None,
        normalize: bool = True,
        allow_dupes: bool = False
    ) -> pd.DataFrame:
        self.check_prompts(prompts, allow_dupes)
        if isinstance(response_seeds, str):
            response_seeds = [response_seeds] * len(prompts)
        elif len(response_seeds) != len(prompts):
            raise ValueError("`prompts` and `response_seeds` must be the same size")
        responses = []
        for prompt, response_seed in zip(prompts, response_seeds):
            responses.append(self.logprobs(prompt, response_seed, allowed_tokens, normalize))
        return pd.DataFrame({"prompt": prompts, "response": responses})

