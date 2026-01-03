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


import math
import numpy as np
import pandas as pd


def pmf_mean(pmf: pd.Series) -> float:
    return sum(index * pmf[index] for index in pmf.index)


def pmf_mode(pmf: pd.Series) -> float:
    return max(pmf.index, key=lambda index: pmf[index])


def pmf_std_dev(pmf: pd.Series) -> float:
    mean = pmf_mean(pmf)
    return math.sqrt(sum(pmf[index] * (index - mean) ** 2 for index in pmf.index))


def softmax(logprobs: pd.Series, temperature: float = 1.0) -> pd.Series:
    logprobs = (logprobs - np.max(logprobs)) / temperature
    probs = pd.Series(np.exp(logprobs)).rename(index={
        index: f"prob_{index}"
        for index in logprobs.index
    })
    return probs / np.sum(probs)

