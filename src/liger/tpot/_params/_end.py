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


from dataclasses import dataclass, field


@dataclass(slots=True)
class EndParams:
    segment_generations: int | None = None
    total_generations: int | None = None
    segment_time: float | None = 60.0
    total_time: float | None = 600.0
    early_stop: int | None = None
    clear_population_file: bool = True
    segment_is_endless_: bool = field(init=False)
    run_is_endless_: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.segment_generations is None:
            self.segment_generations = self.total_generations
        if self.segment_time is None:
            self.segment_time = self.total_time
        self.segment_is_endless_ = self._is_segment_endless()
        self.run_is_endless_ = self._is_run_endless()

    def _is_segment_endless(self) -> bool:
        return (
            self.segment_generations is None
            and
            (self.segment_time is None or self.segment_time == float("inf"))
        )

    def _is_run_endless(self) -> bool:
        return (
            self.total_generations is None
            and
            (self.total_time is None or self.total_time == float("inf"))
            and
            self.early_stop is None
        )
