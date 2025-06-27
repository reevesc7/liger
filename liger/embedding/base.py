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
import re


class BaseEmbedder:
    @staticmethod
    @overload
    def alter_strings(
        strings: str,
        to_replace: str,
        replacement: str
    ) -> str: ...
    @staticmethod
    @overload
    def alter_strings(
        strings: MutableSequence[str] | pd.Series,
        to_replace: str,
        replacement: str
    ) -> pd.Series: ...
    @staticmethod
    def alter_strings(
        strings: str | MutableSequence[str] | pd.Series,
        to_replace: str,
        replacement: str,
    ) -> str | pd.Series:
        """Substitute a string within all strings.

        Parameters
        ----------
        `strings` : `str` or `MutableSequence[str]`
            The strings to be transformed.
        `to_replace` : `str`
            The substring to replace in each `strings` string.
        `add_string` : `str`
            The string to add instead for each `strings` string.

        Returns
        -------
        `strings` : `str` or `list[str]`
            The strings with replacements.
        """
        if isinstance(strings, str):
            return re.sub(to_replace, replacement, strings)
        return pd.Series(
            BaseEmbedder.alter_strings(string, to_replace, replacement)
            for string in strings
        )

    @staticmethod
    @overload
    def prepend_to_strings(strings: str, add_string: str) -> str: ...
    @staticmethod
    @overload
    def prepend_to_strings(
        strings: MutableSequence[str] | pd.Series,
        add_string: str
    ) -> pd.Series: ...
    @staticmethod
    def prepend_to_strings(
        strings: str | MutableSequence[str] | pd.Series,
        add_string: str,
    ) -> str | pd.Series:
        """Add a string to the start of all strings.

        Parameters
        ----------
        `strings` : `str` or `MutableSequence[str]`
            The strings to be transformed.
        `add_string` : `str`
            The string to add to the beginning of each `strings` string.

        Returns
        -------
        `strings` : `str` or `list[str]`
            The strings with `add_string` prepended to each.
        """
        if isinstance(strings, str):
            return add_string.strip() + " " + strings
        return pd.Series(
            BaseEmbedder.prepend_to_strings(string, add_string)
            for string in strings
        )

    def set_model(self, model_str: str) -> None:
        self.model = model_str

    @overload
    def embed(self, strings: str) -> pd.Series: ...
    @overload
    def embed(self, strings: MutableSequence[str] | pd.Series) -> pd.DataFrame: ...
    def embed(self, strings: str | MutableSequence[str] | pd.Series) -> pd.Series | pd.DataFrame:
        raise NotImplementedError()

