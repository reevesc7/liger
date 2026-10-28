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


from enum import StrEnum


class RunCol(StrEnum):
    ID = "run_id"


class IndivCol(StrEnum):
    ID = "indiv_id"
    PARENTS = "Parents"
    VAR_FUNC = "Variation_Function"
    GEN = "Generation"
    SUB_TIME = "Submitted Timestamp"
    COMP_TIME = "Completed Timestamp"
    EVAL_ERR = "Eval Error"
    PARETO = "Pareto_Front"
    INSTANCE = "Instance"
    INDIV = "Individual"


class NodeCol(StrEnum):
    ID = "node_id"
    INSTANCE = "instance"
    SUCCESSORS = "successors"
    TYPE = "node_type"


class EdgeCol(StrEnum):
    SOURCE_ID = "source_id"
