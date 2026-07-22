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


from typing import Any, Callable, Self, TextIO
import warnings
import sys
from pathlib import Path
import shutil
import json
from random import randint
from datetime import datetime, timezone
import warnings
import numpy as np
import pandas as pd
from liger.dataset import Dataset
from liger.training_testing import init_objects
from liger.training.tpot.serde.search_space import create_search_space
from tpot import TPOTEstimator, Population
from sklearn.pipeline import Pipeline
from tpot.graphsklearn import GraphPipeline
from networkx.classes import DiGraph
import dill


warnings.filterwarnings(
    "ignore",
    message="The hashes produced for directed graphs changed in version v3.5",
)
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
)


class TPOTManager:
    OUTPUT = Path("outputs/")
    IN_PROGRESS = Path("in_progress/")
    MANAGER_DATA = Path("manager_data.json")
    POPULATION_PKL = Path("population.pkl")
    TEMP_POPULATION_PKL = Path("temp-population.pkl")
    FITTED_PIPELINE = Path("fitted_pipeline.pkl")
    DATETIME_FMT = "%Y-%m-%d_%H-%M-%S.%f"
    MANAGER_PARAM_KEYS = [
        "id",
        "config_file",
        "data_file",
        "output_dir",
        "feature_keys",
        "target_keys",
        "feature_transformers",
        "target_transformers",
        "feature_transformers_kwargs",
        "target_transformers_kwargs",
        "target_gens",
        "eval_random_states",
        "export_fitted_pipeline",
        "clean_population_file",
    ]
    TPOT_PARAM_KEYS = [
        "search_space",
        "scorers",
        "scorers_weights",
        "classification",
        "cv",
        "other_objective_functions",
        "other_objective_functions_weights",
        "objective_function_names",
        "bigger_is_better",
        "export_graphpipeline",
        "memory",
        "categorical_features",
        "preprocessing",
        "population_size",
        "initial_population_size",
        "population_scaling",
        "generations_until_end_population",
        "generations",
        "max_time_mins",
        "max_eval_time_mins",
        "validation_strategy",
        "validation_fraction",
        "disable_label_encoder",
        "early_stop",
        "scorers_early_stop_tol",
        "other_objectives_early_stop_tol",
        "threshold_evaluation_pruning",
        "threshold_evaluation_scaling",
        "selection_evaluation_pruning",
        "selection_evaluation_scaling",
        "min_history_threshold",
        "survival_percentage",
        "crossover_probability",
        "mutate_probability",
        "mutate_then_crossover_probability",
        "crossover_then_mutate_probability",
        "budget_range",
        "budget_scaling",
        "generations_until_end_budget",
        "stepwise_steps",
        "n_jobs",
        "memory_limit",
        "client",
        "processes",
        "warm_start",
        "periodic_checkpoint_folder",
        "callback",
        "verbose",
        "scatter",
        "random_state",
    ]
    MANAGER_ATTR_KEYS = [
        "complete_gens",
        "gen_scores",
        "segment_start_times",
        "segment_run_times",
        "slurm_ids",
    ]


    def __init__(
        self,
        config_file: str | Path | None = None,
        tpot_random_state: int | None = None,
        slurm_id: int | None = None,
        id: str | None = None,
        manager_parameters: dict | None = None,
        tpot_parameters: dict | None = None,
        manager_attributes: dict | None = None,
    ) -> None:
        self.start_time = datetime.now(timezone.utc)
        self.config_file: str | Path | None = config_file
        _manager_params, _tpot_params, _manager_attrs = self.load_config(self.config_file)

        # Override config parameters with argument parameters
        if isinstance(manager_parameters, dict):
            _manager_params.update(manager_parameters)
        if isinstance(tpot_parameters, dict):
            _tpot_params.update(tpot_parameters)
        if isinstance(manager_attributes, dict):
            _manager_attrs.update(manager_attributes)

        if self.config_file is None:
            self.config_file = _manager_params.get("config_file")

        self.id = self.use_first(
            id,
            _manager_params.get("id"),
            self.start_time.strftime(self.DATETIME_FMT),
        )
        self.data_file: str | Path | None = _manager_params.get("data_file")
        if self.data_file is None:
            raise ValueError("Data file path is unspecified")
        self.data_file = Path(self.data_file)
        self._config_output_dir = _manager_params.get("output_dir", self.data_file.stem)
        self.output_dir = self.OUTPUT / self._config_output_dir / str(self.id)
        self.feature_keys: list[str] | None = _manager_params.get("feature_keys", None)
        self.target_keys: list[str] | None = _manager_params.get("target_keys", None)
        if self.feature_keys is None or self.target_keys is None:
            raise ValueError("Feature keys or target keys are unspecified")
        self.feature_transformers: list[str] | None = _manager_params.get("feature_transformers")
        self.target_transformers: list[str] | None = _manager_params.get("target_transformers")
        self.feature_transformers_kwargs: list[dict[str, Any] | None] | None = _manager_params.get(
            "feature_transformers_kwargs",
            None,
        )
        self.target_transformers_kwargs: list[dict[str, Any] | None] | None = _manager_params.get(
            "target_transformers_kwargs",
            None,
        )
        self.dataset = Dataset.from_csv(
            self.data_file,
            self.feature_keys,
            self.target_keys,
            self.feature_transformers,
            self.target_transformers,
            self.feature_transformers_kwargs,
            self.target_transformers_kwargs,
        )
        if any(dim==0 for dim in self.dataset.x.shape):
            raise ValueError(
                "No data in Dataset.x. Perhaps \"feature_keys\" filter did not "
                "match any columns in the dataset file"
            )
        if any(dim==0 for dim in self.dataset.y.shape):
            raise ValueError(
                "No data in Dataset.y. Perhaps \"target_keys\" filter did not "
                "match any columns in the dataset file"
            )
        self.target_gens: int = _manager_params.get("target_gens", 10)
        _eval_random_states: int | list[int] = _manager_params.get("eval_random_states", 1)
        if isinstance(_eval_random_states, int):
            _eval_random_states = max(1, _eval_random_states)
            self.eval_random_states = [
                randint(0, 2 ** 32 - 1)
                for _ in range(_eval_random_states)
            ]
        else:
            self.eval_random_states = _eval_random_states
        self.export_fitted_pipeline = _manager_params.get("export_fitted_pipeline", True)
        self.clean_population_file = _manager_params.get("clean_population_file", False)

        self._config_search_space = _tpot_params["search_space"]
        self._config_scorers = _tpot_params["scorers"]
        self._config_other_objectives = _tpot_params.get("other_objective_functions", [])
        _tpot_random_state: int = self.use_first(
            tpot_random_state,
            _tpot_params.get("random_state"),
            randint(0, 2 ** 32 - 1),
        )

        self.complete_gens: int = _manager_attrs.get("complete_gens", 0)
        self.gen_scores: list[list[float]] = _manager_attrs.get("gen_scores", [])
        self.segment_start_times: list[str] = _manager_attrs.get("segment_start_times", [])
        self.segment_start_times.append(self.start_time.strftime(self.DATETIME_FMT))
        self.segment_run_times: list[float] = _manager_attrs.get("segment_run_times", [])
        self.slurm_ids: list[int | None] = _manager_attrs.get("slurm_ids", [])
        self.slurm_ids.append(slurm_id)

        unmodified_params: dict[str, Any] = {
            key: value
            for key, value in _tpot_params.items() if key not in [
                "search_space",
                "scorers",
                "cv",
                "other_objective_functions",
                "survival_selector",
                "parent_selector",
                "periodic_checkpoint_folder",
                "random_state",
            ]
        }
        self.tpot = TPOTEstimator(
            search_space=create_search_space(
                self._config_search_space,
                self.dataset.x.shape[0],
                self.dataset.x.shape[1],
                _tpot_random_state
            ),
            scorers=init_objects(self._config_scorers),
            cv=self.get_cv(
                _tpot_params.get("cv"),
                _tpot_params["classification"],
                self.dataset.y
            ),
            other_objective_functions=init_objects(self._config_other_objectives),
            periodic_checkpoint_folder=self.output_dir,
            random_state=_tpot_random_state,
            **unmodified_params,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, slurm_id: int | None) -> Self:
        checkpoint = Path(checkpoint)
        return cls(config_file=checkpoint / cls.MANAGER_DATA, slurm_id=slurm_id)

    @staticmethod
    def use_first(*args) -> Any:
        """Return the first arg which is not None.
        """
        for arg in args:
            if arg is not None:
                return arg
        return None

    @staticmethod
    def find_checkpoint(id: str) -> Path | None:
        for checkpoint_path in TPOTManager.OUTPUT.rglob(id):
            return checkpoint_path
        return None

    @staticmethod
    def load_config(
        config_path: str | Path | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if config_path is None:
            return ({}, {}, {})
        config_path = Path(config_path)
        with open(config_path, "r") as file:
            config = dict(json.load(file))
        manager_parameters = config.get("manager_parameters", {})
        tpot_parameters = config.get("tpot_parameters", {})
        manager_attributes = config.get("manager_attributes", {})
        if not isinstance(manager_parameters, dict):
            raise TypeError(f"manager_parameters should be type dict, not {type(manager_parameters)}")
        if not isinstance(tpot_parameters, dict):
            raise TypeError(f"tpot_parameters should be type dict, not {type(tpot_parameters)}")
        if not isinstance(manager_attributes, dict):
            raise TypeError(f"manager_attributes should be type dict, not {type(manager_attributes)}")
        return (manager_parameters, tpot_parameters, manager_attributes)

    @staticmethod
    def get_cv(param_cv: int | None, classification: bool, y: pd.DataFrame) -> int:
        if classification:
            _, counts = np.unique(y, return_counts=True)
            if counts.size == 1:
                max_cv = int(counts[0])
            else:
                max_cv = int(np.sort(counts)[-2])
        else:
            max_cv = y.shape[0]
        if param_cv is None:
            return max_cv
        elif param_cv > max_cv:
            warnings.warn((
                f"Config \"cv\"={param_cv} is greater than the dataset allows. "
                f"Using max allowed by dataset ({max_cv})"
            ))
            return max_cv
        return param_cv

    @staticmethod
    def callable_to_string(callable: Callable) -> str:
        module = getattr(callable, "__module__", None)
        qualname = getattr(callable, "__qualname__", None)
        name = getattr(callable, "__name__", None)
        if not(module is None or qualname is None):
            return f"{module}.{qualname}"
        elif not(module is None or name is None):
            return f"{module}.{name}"
        elif name is not None:
            return name
        else:
            return repr(callable)

    @staticmethod
    def json_everything(objec: Any) -> Any:
        if isinstance(objec, pd.Series):
            return {index: value for index, value in zip(objec.index, objec.to_list())}
        if isinstance(objec, pd.DataFrame):
            return {
                col: TPOTManager.json_everything(objec[col])
                for col in objec.columns
                if col != "Individual"
            }
        if isinstance(objec, np.ndarray):
            return objec.tolist()
        if isinstance(objec, range):
            return list(objec)
        if isinstance(objec, (Pipeline, GraphPipeline)):
            method = f"{type(objec).__module__}.{type(objec).__name__}"
            return {
                "method": method,
                "params": objec.get_params(deep=False),
            }
        if isinstance(objec, Path):
            return str(objec)
        if isinstance(objec, DiGraph):
            return {key: value for key, value in objec.nodes.items()}
        if hasattr(objec, "__dict__"):
            method = f"{type(objec).__module__}.{type(objec).__name__}"
            return {"method": method, "params": {
                key: value
                for key, value in objec.__dict__.items()
                if not key.startswith("_") and not key.endswith("_")
            }}
        if isinstance(objec, np.generic):
            return objec.item()
        if isinstance(objec, Callable):
            return TPOTManager.callable_to_string(objec)
        raise TypeError(f"Could not convert type {type(objec)} to json format")

    def get_manager_data(self) -> dict:
        manager_parameters = {
            key: self.__dict__.get(key, None)
            for key in self.MANAGER_PARAM_KEYS
        }
        manager_parameters["output_dir"] = self._config_output_dir
        tpot_parameters = {
            key: self.tpot.__dict__.get(key, None)
            for key in self.TPOT_PARAM_KEYS
        }
        tpot_parameters["search_space"] = self._config_search_space
        tpot_parameters["scorers"] = self._config_scorers
        tpot_parameters["other_objective_functions"] = self._config_other_objectives
        manager_attributes = {
            key: self.__dict__.get(key, None)
            for key in self.MANAGER_ATTR_KEYS
        }
        tpot_attributes = {}
        if self.tpot.evaluated_individuals is not None:
            tpot_attributes["fitted_pipeline_id"] = self.tpot.evaluated_individuals[self.tpot.objective_names[0]].idxmax()
            tpot_attributes["evaluated_individuals"] = self.tpot.evaluated_individuals.drop([
                "Individual",
            ], axis=1)
        return {
            "manager_parameters": manager_parameters,
            "tpot_parameters": tpot_parameters,
            "manager_attributes": manager_attributes,
            "tpot_attributes": tpot_attributes,
        }

    def save_data(self) -> None:
        manager_data = self.get_manager_data()
        with open(self.output_dir / self.MANAGER_DATA, "w") as file:
            json.dump(manager_data, file, indent=4, default=self.json_everything)
        with open(self.output_dir / self.POPULATION_PKL, "rb") as file:
            pop: Population = dill.load(file)
        pop.evaluated_individuals["Generation"] = pop.evaluated_individuals["Generation"].astype("Int64")
        with open(self.output_dir / self.POPULATION_PKL, "wb") as file:
            dill.dump(pop, file)

    def cleanup(self) -> None:
        if self.clean_population_file and (self.output_dir / self.POPULATION_PKL).is_file():
            (self.output_dir / self.POPULATION_PKL).unlink()

    def append_scores(self, output_lines: list[str]) -> None:
        gen_indices = [
            index
            for index, line in enumerate(output_lines)
            if "Generation:  " in line
        ] + [len(output_lines)]
        for gen in range(len(gen_indices) - 1):
            gen_start = gen_indices[gen]
            gen_end = gen_indices[gen + 1]
            self.gen_scores.append([
                float(line.split(": ")[-1])
                for line in output_lines[gen_start:gen_end]
                if "score: " in line
            ])

    def update_complete_gens(self, output_lines: list[str]) -> None:
        self.complete_gens = int([
            line
            for line in output_lines
            if "Generation:  " in line
        ][-1].split(":  ")[-1].removesuffix(".0"))

    def run_segment(self) -> None:
        capture = LiveOutputCapture()
        sys.stdout = capture
        self.save_prep()
        self.in_progress()
        print("\nRUN ID:", self.id, flush=True)
        print("TPOT RANDOM STATE:", self.tpot.random_state, flush=True)
        if self.complete_gens >= self.target_gens or self.detect_early_stop():
            self.not_in_progress()
            print("\nRUN TERMINATION CONDITIONS ALREADY MET")
            print("\nRUN COMPLETE")
            return
        self.tpot.fit(self.dataset.x, self.dataset.y)
        output = capture.get_output()
        output_lines = output.split("\n")
        if "Generation:  " not in output and "score: " not in output:
            raise Exception("Fitting ended improperly... quitting")
        self.append_scores(output_lines)
        self.update_complete_gens(output_lines)
        self.run_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        self.segment_run_times.append(self.run_time)
        if self.complete_gens >= self.target_gens or self.detect_early_stop():
            self.export_pipeline()
            self.save_data()
            self.cleanup()
            self.not_in_progress()
            print("\nFITTED PIPELINE:")
            print(self.tpot.fitted_pipeline_)
            print("\nRUN COMPLETE")
            return
        self.save_data()
        self.not_in_progress()
        print(f"\nRUN INCOMPLETE WITH ID: {self.id}")

    def save_prep(self) -> None:
        """Ensure the existence of the output and in-progress directories.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.IN_PROGRESS.mkdir(parents=True, exist_ok=True)

    def in_progress(self):
        """Set up in-progress signals.
        #
        If in-progress signals already exist, use them to recover
        state from the last time this training was run.
        A temporary population file is created before any training
        occurs, which allows for reinstantiation from that file
        if training was cut off mid-segment.
        Additionally, a small file is generated in the in-progress
        directory, enabling detection of runs which are currently
        running or were cut off mid-segment.
        """
        if (self.output_dir / self.TEMP_POPULATION_PKL).is_file():
            print("USING", self.TEMP_POPULATION_PKL, flush=True)
            shutil.copy(
                self.output_dir / self.TEMP_POPULATION_PKL,
                self.output_dir / self.POPULATION_PKL,
            )
        elif (self.output_dir / self.POPULATION_PKL).is_file():
            print("USING", self.POPULATION_PKL, flush=True)
            shutil.copy(
                self.output_dir / self.POPULATION_PKL,
                self.output_dir / self.TEMP_POPULATION_PKL,
            )
        else:
            print("NO PRE-EXISTING POPULATION FILE FOUND - GENERATING POPULATION")
        with open(self.IN_PROGRESS / (str(self.id) + ".txt"), "w") as file:
            file.writelines([
                "Start: UTC " + str(datetime.now(timezone.utc)),
                "\nGeneration: " + str(self.complete_gens + 1),
                "\nSLURM JOB ID: " + str(self.slurm_ids[-1]),
            ])

    def not_in_progress(self) -> None:
        """Clear in-progress signals.
        #
        Remove the temporary population file and in-progress file.
        If training was cut off mid-segment, this method will not
        run, and new segments will initialize with the same state
        as the cut-off segment, enabling re-attempting it.
        """
        if (self.output_dir / self.TEMP_POPULATION_PKL).is_file():
            (self.output_dir / self.TEMP_POPULATION_PKL).unlink()
        if (self.IN_PROGRESS / (str(self.id) + ".txt")).is_file():
            (self.IN_PROGRESS / (str(self.id) + ".txt")).unlink()

    def detect_early_stop(self) -> bool:
        """Detect whether the early stop condition has been met.
        #
        Takes into account the `TPOTEstimator`'s `early_stop` and
        `early_stop_tol` attributes.
        Triggers if there has not been improvement ACROSS `early_stop` generations;
        i.e., the score must be the same for `early_stop` + 1 generations.
        """
        if not isinstance(self.tpot.early_stop, int):
            return False
        if len(self.gen_scores) < self.tpot.early_stop + 1:
            return False
        if any(abs(a_score - z_score) >= tol for a_score, z_score, tol in zip(
            self.gen_scores[-1 - self.tpot.early_stop],
            self.gen_scores[-1],
            self.tpot.early_stop_tol,
        )):
            return False
        return True

    def export_pipeline(self) -> None:
        """Creates a pickle file of the best performing pipeline.
        #
        The file is made in the output directory.
        Pickling pipelines relies on the `dill` module, so loading
        from a fitted pipeline pickle file requires `dill`.
        """
        if not self.export_fitted_pipeline:
            return
        with open(self.output_dir / self.FITTED_PIPELINE, "wb") as file:
            dill.dump(self.tpot.fitted_pipeline_, file)

class LiveOutputCapture:
    """A stand-in for `sys.stdout` which records the text it writes.
    #
    Recorded text can be retrieved with the `get_output()` method.
    """
    def __init__(self):
        self.captured_text = []
        self.original_stdout: TextIO = sys.stdout

    def write(self, text: str) -> int:
        """Record the written text before writing it, as normal.
        """
        self.captured_text.append(text)
        return self.original_stdout.write(text)

    def flush(self) -> None:
        """Flush print buffer, as normal.
        """
        self.original_stdout.flush()

    def get_output(self) -> str:
        """Retreive all recorded text as a string.
        """
        return "".join(self.captured_text)

