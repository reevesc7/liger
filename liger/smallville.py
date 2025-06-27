from pathlib import Path
import pandas as pd


PROMPT_START = "@PROMPT"
PROMPT_END = "@RESPONSE"


def get_logged_prompts(file_path: str | Path, filter: str) -> pd.Series:
    file_path = Path(file_path)
    with open(file_path, "r", encoding="cp1252") as file:
        lines = file.readlines()
    prompts: list[str] = []
    start_index = 0
    for index, line in enumerate(lines):
        if PROMPT_START in line:
            start_index = index + 1
            continue
        elif PROMPT_END in line:
            prompt = "".join(lines[start_index:index])
            if filter not in prompt:
                continue
            prompts.append(prompt)
            continue
    return pd.Series(prompts, name="prompt")


def response_strip(responses: pd.Series) -> pd.Series:
    """Convert full Smallville poignancy responses to ints.

    Parameters
    ----------
    `responses` : `pandas.Series`
        The "response" column returned by `OpenAISurveyor. generate_response(),
        .generate_responses()`, `.survey()`. Elements are either `str`
        or `list[str]`, depending on number of replicates.

    Returns
    -------
    `stripped_responses` : `pandas.Series`
        The "response" column, with each element formatted as `int` if inputs
        were `str` or `list[int]` is inputs were `list[str]`.
    """
    if isinstance(responses[0], str):
        stripped = [
            int(rating.split(":")[-1].strip("}{\"\n\" "))
            for rating in responses
        ]
    else:
        stripped = [
            [int(rating.split(":")[-1].strip("}{\"\n\" ")) for rating in ratings]
            for ratings in responses
        ]
    return pd.Series(data=stripped, name=responses.name)

