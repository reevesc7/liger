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


import pandas as pd


class BaseSurveyor:
    @staticmethod
    def check_prompts(prompts: list[str], allow_dupes: bool = False) -> None:
        if not allow_dupes and len(set(prompts)) < len(prompts):
            raise ValueError("Duplicate prompts detected. Set `allow_dupes = True` to ignore")


    def survey(self, prompts: list[str], reps: int = 1) -> pd.DataFrame:
        raise NotImplementedError()

