from typing import Literal, NotRequired, TypedDict
import sys
from dataclasses import dataclass
from pathlib import Path
import tomllib
import pandas as pd
from openai import OpenAI


DIR = Path(__file__).parent.resolve()


class TextConfig(TypedDict):
    type: Literal["text"]
    temperature: float
    column: str


class LogprobsConfig(TypedDict):
    type: Literal["logprobs"]
    allowed_tokens: NotRequired[list[str]]
    floor_margin: NotRequired[float]


class CSVColumn(TypedDict):
    path: str
    column: str | None


class ConfigMessage(TypedDict):
    role: str
    content: str | CSVColumn


@dataclass(slots=True)
class Config:
    key_path: str
    model: str
    result_path: str
    messages: pd.DataFrame
    response: TextConfig | LogprobsConfig

    def __init__(
        self,
        key_path: str,
        model: str,
        result_path: str,
        messages: list[ConfigMessage],
        response: TextConfig | LogprobsConfig,
    ) -> None:
        self.key_path = key_path
        self.model = model
        self.result_path = result_path
        self.response = response
        contents: list[dict[str, str] | pd.Series] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                contents.append({"role": message["role"], "content": content})
                continue
            contents.append(pd.Series(self._read_column(**content).apply(
                lambda el: {"role": message["role"], "content": el}
            )))
        n_rows = max(col.size if isinstance(col, pd.Series) else 1 for col in contents)
        full_contents = [
            col if isinstance(col, pd.Series) else [col] * n_rows for col in contents
        ]
        self.messages = pd.DataFrame(
            {col: content for col, content in enumerate(full_contents)}
        )

    @staticmethod
    def _read_column(path: str, column: str | None = None) -> pd.Series:
        if column is None:
            return pd.Series(pd.read_csv(DIR / path).iloc[:, 0])
        return pd.Series(pd.read_csv(
            DIR / path,
            usecols=lambda col: col == column,
        )[column])


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
    cfg = Config(**tomllib.load(Path(sys.argv[1]).open("rb")))
    client = OpenAI(api_key=(DIR / cfg.key_path).read_text().strip())
    responses = pd.DataFrame(
        cfg.messages.apply(
            lambda row: _response(cfg, client, row),
            result_type="expand",
            axis=1,
        )
    )
    if cfg.response["type"] == "text":
        responses.columns = [cfg.response["column"]]
    responses.to_csv(DIR / cfg.result_path, index=False)


if __name__ == "__main__":
    main()
