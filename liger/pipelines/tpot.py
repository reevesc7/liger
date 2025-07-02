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


from types import FunctionType
from typing import Any
import sys
from pathlib import Path
from copy import deepcopy
import shutil
import json
from random import randint
from datetime import datetime, timezone
from importlib import import_module
from typing_extensions import LiteralString, Self
import numpy as np
import pandas as pd
from ..dataset import Dataset
from ..training_testing import kfold_predict
from ..search_space_creator import create_search_space
from tpot import TPOTEstimator
from sklearn.metrics._scorer import _Scorer
from sklearn.pipeline import Pipeline
import dill


class TPOTPipeline:
    OUTPUT = Path("outputs/")
    IN_PROGRESS = Path("in_progress/")
    PIPELINE_DATA = Path("pipeline_data.json")
    POPULATION_PKL = Path("population.pkl")
    TEMP_POPULATION_PKL = Path("temp-population.pkl")
    FITTED_PIPELINE = Path("fitted_pipeline.pkl")
    DATETIME_FMT = "%Y-%m-%d_%H-%M-%S.%f"
    PIPELINE_PARAM_KEYS = {
        "config_file",
        "data_file",
        "feature_keys",
        "score_keys",
        "target_gens",
        "eval_random_states",
        "id",
    }
    TPOT_PARAM_KEYS = {
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
    }
    PIPELINE_ATTR_KEYS = {
        "complete_gens",
        "gen_scores",
        "sxn_start_times",
        "sxn_run_times",
        "slurm_ids",
        "kfold_scores",
        "kfold_predictions",
    }
    TPOT_ATTR_KEYS = {
        "fitted_pipeline_",
        "evaluated_individuals",
        "pareto_front",
    }


    def __init__(
        self,
        config_file: str | None = None,
        tpot_random_state: int | None = None,
        slurm_id: int | None = None,
        id: str | None = None,
        pipeline_parameters: dict | None = None,
        tpot_parameters: dict | None = None,
        pipeline_attributes: dict | None = None,
    ) -> None:
        self.config_file: str | None = config_file
        _pipeline_params, _tpot_params, _pipeline_attrs = self.load_config(self.config_file)

        # Override config parameters with argument parameters
        if isinstance(pipeline_parameters, dict):
            _pipeline_params.update(pipeline_parameters)
        if isinstance(tpot_parameters, dict):
            _tpot_params.update(tpot_parameters)
        if isinstance(pipeline_attributes, dict):
            _pipeline_attrs.update(pipeline_attributes)

        # Record start time
        self.start_time = datetime.now(timezone.utc)

        # Set special needs parameters
        if self.config_file is None:
            self.config_file = _pipeline_params.get("config_file")
        self.data_file: str | None = _pipeline_params.get("data_file", None)
        self.feature_keys: list[str] | None = _pipeline_params.get("feature_keys", None)
        self.score_keys: list[str] | None = _pipeline_params.get("score_keys", None)
        if self.data_file is None or self.feature_keys is None or self.score_keys is None:
            raise ValueError("Must specify a data file and feature and score keys in config")
        self.tpot_random_state: int = self.use_first(
            tpot_random_state,
            _tpot_params.get("random_state"),
            randint(0, 2**32-1),
        )
        self.target_gens: int = _pipeline_params.get("target_gens", 10)
        self.eval_random_states: list[int] = _pipeline_params.get("eval_random_states", [0])
        self.id = self.use_first(
            id,
            _pipeline_params.get("id"),
            self.start_time.strftime(self.DATETIME_FMT),
        )
        self.complete_gens: int = _pipeline_attrs.get("complete_gens", 0)
        self.gen_scores: list[list[float]] = _pipeline_attrs.get("gen_scores", [])
        self.sxn_start_times: list[str] = _pipeline_attrs.get("sxn_start_times", [])
        self.sxn_start_times.append(self.start_time.strftime(self.DATETIME_FMT))
        self.sxn_run_times: list[float] = _pipeline_attrs.get("sxn_run_times", [])
        self.slurm_ids: list[int | None] = _pipeline_attrs.get("slurm_ids", [])
        self.slurm_ids.append(slurm_id)
        self.kfold_scores: dict = _pipeline_attrs.get("kfold_scores", {})
        self.kfold_predictions: dict = _pipeline_attrs.get("kfold_predictions", {})

        # Set dataset
        self.data_name = self.get_filename(self.data_file)
        self.dataset = Dataset.from_csv(self.data_file, self.feature_keys, self.score_keys)

        # Set CV
        if _tpot_params["classification"]:
            _, counts = np.unique(self.dataset.y, return_counts=True)
            if counts.size == 1:
                max_cv = int(counts[0])
            else:
                max_cv = int(np.sort(counts)[-2])
        else:
            max_cv = self.dataset.y.shape[0]
        cv = _tpot_params.get("cv")
        if cv is None:
            cv = max_cv
        elif cv > max_cv:
            print(f"WARNING: Config \"cv\"={cv} is greater than the dataset allows. Using max allowed by dataset ({max_cv})", flush=True)
            cv = max_cv
        else:
            cv = int(_tpot_params["cv"])

        # Create search space
        self.config_search_space = _tpot_params["search_space"]
        search_space = create_search_space(
            self.config_search_space,
            self.dataset.x.shape[1],
            self.tpot_random_state
        )

        # Set scorer functions
        scorers = self.init_scorers(_tpot_params["scorers"])

        # Set output path
        self.output_dir = self.OUTPUT / self.data_name / str(self.id)

        # Override mid-generation population.pkl, if need be
        # Create on-generation temp-population.pkl, if nonexistent
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

        # Initialize estimator
        self.tpot = TPOTEstimator(
            search_space=search_space,
            scorers=scorers,
            cv=cv,
            periodic_checkpoint_folder=self.output_dir,
            random_state=self.tpot_random_state,
            **{key: value for key, value in _tpot_params.items() if key not in [
                "search_space",
                "scorers",
                "cv",
                "survival_selector",
                "parent_selector",
                "periodic_checkpoint_folder",
                "random_state",
            ]},
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, slurm_id: int | None) -> Self:
        checkpoint = Path(checkpoint)
        pipeline_params, tpot_params, pipeline_attrs = cls.load_config(checkpoint / cls.PIPELINE_DATA)
        kwargs = {
            "pipeline_parameters": pipeline_params,
            "tpot_parameters": tpot_params,
            "pipeline_attributes": pipeline_attrs,
            "slurm_id": slurm_id,
        }
        pipeline = cls(**kwargs)
        return pipeline

    @staticmethod
    def use_first(*args) -> Any:
        """Return the first arg which is not None."""
        for arg in args:
            if arg is not None:
                return arg
        return None

    @staticmethod
    def get_filename(filepath: str | Path) -> str:
        filepath = Path(filepath)
        return filepath.stem

    @staticmethod
    def find_checkpoint(id: str) -> Path | None:
        for checkpoint_path in TPOTPipeline.OUTPUT.rglob(id):
            return checkpoint_path
        return None

    @staticmethod
    def load_config(config_path: str | Path | None) -> tuple[dict, dict, dict]:
        if config_path is None:
            return ({}, {}, {})
        config_path = Path(config_path)
        with open(config_path) as f:
            config = dict(json.load(f))
        pipeline_parameters = config.get("pipeline_parameters", {})
        tpot_parameters = config.get("tpot_parameters", {})
        pipeline_attributes = config.get("pipeline_attributes", {})
        if not isinstance(pipeline_parameters, dict):
            raise TypeError(f"pipeline_parameters should be type dict, not {type(pipeline_parameters)}")
        if not isinstance(tpot_parameters, dict):
            raise TypeError(f"tpot_parameters should be type dict, not {type(tpot_parameters)}")
        if not isinstance(pipeline_attributes, dict):
            raise TypeError(f"pipeline_attributes should be type dict, not {type(pipeline_attributes)}")
        return (pipeline_parameters, tpot_parameters, pipeline_attributes)

    @staticmethod
    def init_scorers(param_scorers: list[str]) -> list[str | FunctionType]:
        scorers: list[str | FunctionType] = []
        for param_scorer in param_scorers:
            if "." not in param_scorer:
                scorers.append(param_scorer)
                continue
            split_scorer = param_scorer.rsplit(".", 1)
            scorers.append(getattr(import_module(split_scorer[0]), split_scorer[1]))
        return scorers

    @staticmethod
    def json_everything(objec: Any) -> Any:
        if isinstance(objec, pd.Series):
            return {index: value for index, value in enumerate(objec.to_list())}
        if isinstance(objec, pd.DataFrame):
            return {
                col: TPOTPipeline.json_everything(objec[col])
                for col in objec.columns
                if col != "Individual"
            }
        if isinstance(objec, np.ndarray):
            return objec.tolist()
        if isinstance(objec, range):
            return list(objec)
        if isinstance(objec, _Scorer):
            return ".".join([objec._score_func.__module__, objec._score_func.__name__])
        if isinstance(objec, FunctionType) and hasattr(objec, "__module__"):
            return ".".join([objec.__module__, objec.__name__])
        if isinstance(objec, Pipeline):
            return objec.__repr__().split("\n")
        if hasattr(objec, "__dict__"):
            return {
                key: value
                for key, value in objec.__dict__.items()
                if not key.startswith("_") and not key.endswith("_")
            }
        return ""

    def save_data(self) -> None:
        pipeline_data = self.get_pipeline_data()
        with open(self.output_dir / self.PIPELINE_DATA, "w") as f:
            json.dump(pipeline_data, f, indent=4, default=self.json_everything)

    def get_pipeline_data(self) -> dict:
        pipeline_parameters = {
            key: value
            for key, value in self.__dict__.items()
            if key in self.PIPELINE_PARAM_KEYS
        }
        tpot_parameters = {
            key: value
            for key, value in self.tpot.__dict__.items()
            if key in self.TPOT_PARAM_KEYS
        }
        tpot_parameters["search_space"] = self.config_search_space
        pipeline_attributes = {
            key: value
            for key, value in self.__dict__.items()
            if key in self.PIPELINE_ATTR_KEYS
        }
        tpot_attributes = {
            key: value
            for key, value in self.tpot.__dict__.items()
            if key in self.TPOT_ATTR_KEYS
        }
        return {
            "pipeline_parameters": pipeline_parameters,
            "tpot_parameters": tpot_parameters,
            "pipeline_attributes": pipeline_attributes,
            "tpot_attributes": tpot_attributes,
        }

    def append_scores(self, output_lines: list[LiteralString]) -> None:
        gen_indices = [
            index
            for index, line in enumerate(output_lines)
            if "Generation:  " in line
        ] + [len(output_lines)]
        for gen in range(len(gen_indices) - 1):
            gen_start = gen_indices[gen]
            gen_end = gen_indices[gen + 1]
            self.gen_scores.append([
                float(l.split(": ")[-1])
                for l in output_lines[gen_start:gen_end]
                if "score: " in l
            ])

    def update_complete_gens(self, output_lines: list[LiteralString]) -> None:
        self.complete_gens = int([
            l
            for l in output_lines
            if "Generation:  " in l
        ][-1].split(":  ")[-1].removesuffix(".0"))

    def run_1_gen(self) -> None:
        capture = LiveOutputCapture()
        sys.stdout = capture
        self.save_prep()
        self.in_progress()
        print("\nPIPELINE ID:", self.id, flush=True)
        print("TPOT RANDOM STATE:", self.tpot.random_state, flush=True)
        if self.complete_gens >= self.target_gens or self.detect_early_stop():
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
        self.sxn_run_times.append(self.run_time)
        if (self.output_dir / self.TEMP_POPULATION_PKL).is_file():
            (self.output_dir / self.TEMP_POPULATION_PKL).unlink()
        if self.complete_gens >= self.target_gens or self.detect_early_stop():
            self.export_fitted_pipeline()
            self.evaluate()
            self.save_data()
            self.not_in_progress()
            print("\nFITTED PIPELINE:")
            print(self.tpot.fitted_pipeline_)
            print("\nRUN COMPLETE")
            return
        self.save_data()
        self.not_in_progress()
        print(f"\nRUN INCOMPLETE WITH ID: {self.id}")

    def save_prep(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.IN_PROGRESS.mkdir(parents=True, exist_ok=True)

    def in_progress(self):
        with open((self.IN_PROGRESS / str(self.id)).with_suffix(".txt"), "w") as f:
            f.writelines([
                "Start: UTC " + str(datetime.now(timezone.utc)),
                "\nGeneration: " + str(self.complete_gens + 1),
                "\nSLURM JOB ID: " + str(self.slurm_ids[-1]),
            ])

    def not_in_progress(self) -> None:
        (self.IN_PROGRESS / str(self.id)).with_suffix(".txt").unlink()

    def detect_early_stop(self) -> bool:
        if not isinstance(self.tpot.early_stop, int):
            return False
        if len(self.gen_scores) < self.tpot.early_stop + 1:
            return False
        if len(set(str(s) for s in self.gen_scores[-(self.tpot.early_stop + 1):])) > 1:
            return False
        return True

    def export_fitted_pipeline(self) -> None:
        with open(self.output_dir / self.FITTED_PIPELINE, "wb") as f:
            dill.dump(self.tpot.fitted_pipeline_, f)

    def evaluate(self) -> None:
        # training_score = self.tpot.score(self.dataset.x, self.dataset.y)
        for eval_random_state in self.eval_random_states:
            kfold_predictions, kfold_scores = self.tpot_test(eval_random_state)
            self.kfold_scores[eval_random_state] = kfold_scores
            self.kfold_predictions[eval_random_state] = kfold_predictions

    def tpot_test(self, eval_random_state: int) -> tuple[list[dict[int, Any]], list[Any]]:
        #set_param_recursive(self.tpot.fitted_pipeline_.steps, 'random_state', eval_random_state)
        #kfold = KFold(n_splits=self.tpot.cv, shuffle=True, random_state=eval_random_state)
        kfold = deepcopy(self.tpot.cv_gen)
        kfold.random_state = eval_random_state
        kfold_predictions, kfold_scores = kfold_predict(
            self.tpot.fitted_pipeline_,
            kfold,
            self.tpot._scorers,
            self.dataset,
        )
        return kfold_predictions, kfold_scores


class LiveOutputCapture:
    def __init__(self):
        self.captured_text = []
        self.original_stdout = sys.stdout

    def write(self, text):
        self.captured_text.append(text)  # Store the output
        self.original_stdout.write(text)  # Still print it to the console

    def flush(self):
        self.original_stdout.flush()  # Ensure flushing still works

    def get_output(self):
        return "".join(self.captured_text)

