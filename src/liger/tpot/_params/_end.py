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


from dataclasses import dataclass
from typing import Any


class _TruthyInt(int):
    """Part of a fix for an off-by-one error in TPOT's early stop logic.
    Always returning true allows TPOT's early stop check to occur when
    `TPOTEstimator.early_stop = 0`, allowing for early stopping in the
    case of 2 generations in a row with the same historical top scores.
    """
    def __bool__(self) -> bool:
        return True


@dataclass(slots=True, frozen=True)
class EndParams:
    segment_gens: int | None = None
    total_gens: int | None = None
    segment_time: float | None = 60.0
    total_time: float | None = 600.0
    early_stop: int | None = None
    clear_population_file: bool = True

    @property
    def effective_segment_gens(self) -> int | None:
        return self.total_gens if self.segment_gens is None else self.segment_gens

    @property
    def effective_segment_time(self) -> float | None:
        return self.total_time if self.segment_time is None else self.segment_time

    @property
    def segment_is_endless(self) -> bool:
        segment_time = self.effective_segment_time
        return (
            self.effective_segment_gens is None
            and (segment_time is None or segment_time == float("inf"))
        )

    @property
    def run_is_endless(self) -> bool:
        return (
            self.total_gens is None
            and (self.total_time is None or self.total_time == float("inf"))
            and self.early_stop is None
        )

    @property
    def tpot_early_stop(self) -> _TruthyInt | None:
        # /// --- Corrects off-by-one error in TPOT's early_stop logic --- \\\
        return None if self.early_stop is None else _TruthyInt(self.early_stop - 1)
        # \\\ ------------------------------------------------------------ ///

    def tpot_kwargs(self) -> dict[str, Any]:
        return {
            "generations": self.effective_segment_gens,
            "max_time_mins": self.effective_segment_time,
            "early_stop": self.tpot_early_stop,
        }
