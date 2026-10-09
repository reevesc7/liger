from typing import Literal, NotRequired, TypedDict
import sys
import contextlib
from dataclasses import dataclass
from pathlib import Path
import tomllib
import pandas as pd
from openai import OpenAI
from liger import dataset as ds


class TextConfig(TypedDict):
    type: Literal["text"]
    temperature: float
    result_column: str


class LogprobsConfig(TypedDict):
    type: Literal["logprobs"]
    allowed_tokens: NotRequired[list[str]]
    floor_margin: NotRequired[float]


class ConfigMessage(TypedDict):
    role: str
    content: str | dict


@dataclass(slots=True)
class Config:
    result_path: Path
    api_key: str
    model: str
    messages: pd.DataFrame
    response: TextConfig | LogprobsConfig

    def __init__(self, cfg_path: Path) -> None:
        self.result_path = cfg_path.with_suffix(".csv")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.api_key = Path(parsed["key_path"]).read_text().strip()
        self.model = parsed["model"]
        self.messages = self._init_messages(parsed["messages"])
        self.response = parsed["response"]

    @staticmethod
    def _init_messages(messages: list[ConfigMessage]) -> pd.DataFrame:
        contents: list[dict[str, str] | pd.Series] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                contents.append({"role": message["role"], "content": content})
                continue
            contents.append(ds.data_from_csv(**content).iloc[:, 0].apply(
                lambda el: {"role": message["role"], "content": el}
            ))
        n_rows = max(col.size if isinstance(col, pd.Series) else 1 for col in contents)
        full_contents = [
            col if isinstance(col, pd.Series) else [col] * n_rows for col in contents
        ]
        return pd.DataFrame(
            {col: content for col, content in enumerate(full_contents)}
        )


def _response(
    cfg: Config,
    client: OpenAI,
    messages: pd.Series,
) -> str | dict[str, float]:
    print("\nRESPONDING TO:")
    print(messages)
    if cfg.response["type"] == "text":
        completion = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.response["temperature"],
        )
        response = completion.choices[0].message.content
        if response is None:
            raise ValueError("No response was returned")
        return response
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
    )
    logprobs = completion.choices[0].logprobs
    if logprobs is not None and logprobs.content is not None:
        top_logprobs = {
            logprob.token: logprob.logprob
            for logprob in logprobs.content[0].top_logprobs
        }
    else:
        raise ValueError("No response was returned or response was unparseable")
    allowed_tokens = cfg.response.get("allowed_tokens")
    if allowed_tokens is not None:
        floor_logprob = min(top_logprobs.values()) - cfg.response.get(
            "floor_margin",
            1.0,
        )
        top_logprobs = {
            f"response_logprob_{token}": top_logprobs.get(token, floor_logprob)
            for token in allowed_tokens
        }
    return top_logprobs


def main():
    cfg_path = Path(sys.argv[1])
    with contextlib.chdir(cfg_path.parent):
        cfg = Config(Path(cfg_path.name))
        client = OpenAI(api_key=cfg.api_key)
        responses = pd.DataFrame(
            cfg.messages.apply(
                lambda row: _response(cfg, client, row),
                result_type="expand",
                axis=1,
            )
        )
        if cfg.response["type"] == "text":
            responses.columns = [cfg.response["result_column"]]
        responses.to_csv(cfg.result_path, index=False)


if __name__ == "__main__":
    main()
