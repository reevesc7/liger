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
from pathlib import Path
from joblib import Memory
from typing import Any, Literal
from distributed import Client


@dataclass(slots=True, frozen=True)
class RuntimeParams:
    n_jobs: int = 1
    memory_limit: Literal["auto"] | str | float | None = "auto"
    client: Client | None = None
    processes: bool = True
    scatter: bool = True
    cache_evaluations: bool = False

    def make_eval_cache(self, output_dir: str | Path) -> Memory | None:
        output_dir = Path(output_dir).resolve()
        if self.cache_evaluations:
            return Memory(output_dir, verbose=0)
        return None

    def tpot_kwargs(self, output_dir: str | Path) -> dict[str, Any]:
        return {
            "memory": self.make_eval_cache(output_dir),
            "n_jobs": self.n_jobs,
            "memory_limit": self.memory_limit,
            "client": self.client,
            "processes": self.processes,
            "scatter": self.scatter,
        }
