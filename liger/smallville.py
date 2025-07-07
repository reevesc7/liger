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
import pandas as pd


def _offset_of_match(line: str, substrs: dict[str, int]) -> int | None:
    for substr, line_offset in substrs.items():
        if substr in line:
            return line_offset


def get_logged_prompts(
    file_path: str | Path,
    start: dict[str, int],
    end: dict[str, int],
    filter: str | set[str],
) -> pd.Series:
    file_path = Path(file_path)
    with open(file_path, "r", encoding="cp1252") as file:
        lines = file.readlines()
    if isinstance(filter, str):
        filter = {filter,}
    prompts: list[str] = []
    start_index = 0
    collecting = False
    for index, line in enumerate(lines):
        if not collecting:
            start_offset = _offset_of_match(line, start)
            if start_offset is None:
                continue
            collecting = True
            start_index = index + start_offset
        else:
            end_offset = _offset_of_match(line, end)
            if end_offset is None:
                continue
            collecting = False
            prompt = "".join(lines[start_index:index + end_offset + 1])
            if not any(substr in prompt for substr in filter):
                continue
            prompts.append(prompt)
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

