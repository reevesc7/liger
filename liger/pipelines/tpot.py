from typing import Any, Union, Callable
import sys
from os import path, makedirs, remove, walk
import json
from random import randint
from datetime import datetime, timezone
from importlib import import_module
import numpy as np
from ..dataset import Dataset
from ..training_testing import kfold_predict
from ..search_space_creator import create_search_space
from tpot import TPOTEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import dill


OUTPUT = "Outputs/"
EXPORT = "Exports/"
CHECKPOINT = "Checkpoints/"
IN_PROGRESS = "InProgress/"
REG_CLASS_OVERLAP = {
    "sklearn.preprocessing.Binarizer",
    "sklearn.decomposition.FastICA",
    "sklearn.cluster.FeatureAgglomeration",
    "sklearn.preprocessing.MaxAbsScaler",
    "sklearn.preprocessing.MinMaxScaler",
    "sklearn.preprocessing.Normalizer",
    "sklearn.kernel_approximation.Nystroem",
    "sklearn.decomposition.PCA",
    "sklearn.preprocessing.PolynomialFeatures",
    "sklearn.kernel_approximation.RBFSampler",
    "sklearn.preprocessing.RobustScaler",
    "sklearn.preprocessing.StandardScaler",
    "tpot.builtins.ZeroCount",
    "tpot.builtins.OneHotEncoder",
    "sklearn.feature_selection.SelectFwe",
    "sklearn.feature_selection.SelectPercentile",
    "sklearn.feature_selection.VarianceThreshold",
    "sklearn.feature_selection.SelectFromModel",
}
PIPELINE_PARAM_KEYS = {
    "config_file",
    "data_file",
    "target_gens",
    "eval_random_states",
    "slurm_id",
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
TPOT_ATTR_KEYS = {
    "evaluated_individuals",
    "fitted_pipeline_"
    "pareto_front",
}


class TPOTPipeline:

    def __init__(
        self,
        config_file: str,
        data_file: str | None = None,
        tpot_random_state: int | None = None,
        slurm_id: int | None = None,
        id: str | None = None,
        complete_gens: int | None = None,
        gen_scores: list[list[float]] | None = None,
    ) -> None:
        self.config_file = config_file
        self.slurm_id = slurm_id
        if complete_gens is not None:
            self.complete_gens = complete_gens
        else:
            self.complete_gens = 0
        if gen_scores is not None:
            self.gen_scores = gen_scores
        else:
            self.gen_scores = []

        # Load config file
        with open(config_file) as f:
            config = dict(json.load(f))
        pipeline_parameters = config.get("pipeline_parameters")
        if pipeline_parameters is None:
            pipeline_parameters = {}
        tpot_parameters = config.get("tpot_parameters")
        if tpot_parameters is None:
            tpot_parameters = {}

        # Record start time
        dt = datetime.now(timezone.utc)
        start_time = dt.strftime("%Y-%m-%d_%H-%M-%S.%f")

        # Set special needs parameters
        self.data_file = TPOTPipeline.use_first(
            data_file,
            pipeline_parameters.get("data_file"),
        )
        if self.data_file is None:
            raise ValueError("No data file was specified")
        self.tpot_random_state = TPOTPipeline.use_first(
            tpot_random_state,
            tpot_parameters.get("random_state"),
            randint(0, 2**32-1),
        )
        self.target_gens = pipeline_parameters["target_gens"]
        self.eval_random_states = pipeline_parameters["eval_random_states"]
        self.id = TPOTPipeline.use_first(
            id,
            pipeline_parameters.get("id"),
            start_time,
        )

        # Set dataset
        self.data_name = TPOTPipeline.get_filename(self.data_file)
        self.dataset = Dataset.from_csv(self.data_file)

        # Set CV
        if tpot_parameters["classification"]:
            _, counts = np.unique(self.dataset.y, return_counts=True)
            if counts.size == 1:
                max_cv = int(counts[0])
            else:
                max_cv = int(np.sort(counts)[-2])
        else:
            max_cv = self.dataset.y.shape[0]
        cv = tpot_parameters.get("cv")
        if cv is None:
            cv = max_cv
        elif cv > max_cv:
            print(f"WARNING: Config \"cv\"={cv} is greater than the dataset allows. Using max allowed by dataset ({max_cv})", flush=True)
            cv = max_cv
        else:
            cv = int(tpot_parameters["cv"])

        # Create search space
        self.config_search_space = tpot_parameters["search_space"]
        search_space = create_search_space(self.config_search_space, self.dataset.X.shape[1], self.tpot_random_state)

        # Set scorer functions
        scorers = TPOTPipeline.init_scorers(tpot_parameters["scorers"])

        # Set output paths
        self.output_dir = OUTPUT + self.data_name + "/"
        self.export_dir = EXPORT + self.data_name + "/"
        self.checkpoint_dir = CHECKPOINT + self.data_name + "/" + self.id + "/"
        self.inprogress_dir = IN_PROGRESS + self.data_name + "/"

        # Initialize estimator
        self.tpot = TPOTEstimator(
            search_space=search_space,
            scorers=scorers,
            cv=cv,
            periodic_checkpoint_folder=self.checkpoint_dir,
            random_state=self.tpot_random_state,
            **{key: value for key, value in tpot_parameters.items() if key not in [
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
    def from_checkpoint(cls, checkpoint_file: str) -> 'TPOTPipeline':
        with open(checkpoint_file, "r") as f:
            kwargs = json.load(f)
            return TPOTPipeline(**kwargs)


    @staticmethod
    def use_first(*args) -> Any:
        """Return the first arg which is not None."""
        for arg in args:
            if arg is not None:
                return arg
        return None


    @staticmethod
    def get_filename(data_file: str) -> str:
        return data_file.rsplit("/")[-1].split(".")[0]


    @staticmethod
    def find_checkpoint(id: str) -> str | None:
        for dirpath, dirnames, _ in walk(CHECKPOINT):
            if id in dirnames and path.isfile(path.join(dirpath, id, "tpot_pipeline.json")):
                return path.join(dirpath, id, "tpot_pipeline.json")
        return None


    @staticmethod
    def init_scorers(param_scorers: list[str]) -> list[Union[str, Callable]]:
        scorers: list[Union[str, Callable]] = []
        for param_scorer in param_scorers:
            if "." not in param_scorer:
                scorers.append(param_scorer)
                continue
            split_scorer = param_scorer.rsplit(".", 1)
            scorers.append(getattr(import_module(split_scorer[0]), split_scorer[1]))
        return scorers


    @staticmethod
    def dict_everything(objec: Any) -> str | None:
        if isinstance(objec, np.ndarray):
            objec = objec.tolist()
            return json.dumps(objec, indent=4, default=TPOTPipeline.dict_everything)
        elif isinstance(objec, range):
            objec = [i for i in objec]
            return json.dumps(objec, indent=4, default=TPOTPipeline.dict_everything)
        elif isinstance(objec, Callable):
            return ".".join([objec.__module__, objec.__name__])
        elif hasattr(objec, "__dict__"):
            return json.dumps(objec.__dict__, indent=4, default=TPOTPipeline.dict_everything)
        else:
            return ""


    def create_checkpoint(self) -> None:
        pipeline_data = self.get_pipeline_data()
        with open(self.checkpoint_dir + "tpot_pipeline.json", "w") as f:
            json.dump(pipeline_data, f, indent=4)


    def get_pipeline_data(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key in [
            "config_file",
            "data_file",
            "tpot_random_state",
            "slurm_id",
            "id",
            "complete_gens",
            "gen_scores",
        ]}


    def run_1_gen(self) -> None:
        # Capture output dynamically
        capture = LiveOutputCapture()
        sys.stdout = capture

        # Save prep
        self.save_prep()
        self.in_progress()
        print("\nPIPELINE ID:", self.id, flush=True)
        print("TPOT RANDOM STATE:", self.tpot.random_state, flush=True)
        output = capture.get_output()
        if self.complete_gens < self.target_gens:
            self.tpot.fit(self.dataset.X, self.dataset.y)
        output = capture.get_output()
        output_lines = output.split("\n")
        if "score: " in output:
            self.gen_scores.append([float(l.split(": ")[-1]) for l in output_lines if "score: " in l])
        if "Generation:  " in output:
            self.complete_gens = int([l for l in output_lines if "Generation:  " in l][0].split(":  ")[-1].strip(".0"))
        else:
            self.complete_gens += 1
        if self.complete_gens >= self.target_gens or self.detect_early_stop():
            self.export_fitted_pipeline()
            self.evaluate()
            self.create_checkpoint()
            self.not_in_progress()
            print("\nFITTED PIPELINE:")
            print(self.tpot.fitted_pipeline_)
            print("\nRUN COMPLETE")
            return
        self.create_checkpoint()
        self.not_in_progress()
        print(f"\nRUN INCOMPLETE WITH ID: {self.id}")
        return


    def save_prep(self) -> None:
        if not path.isdir(self.output_dir):
            makedirs(self.output_dir)
        if not path.isdir(self.export_dir):
            makedirs(self.export_dir)
        if not path.isdir(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)
        if not path.isdir(self.inprogress_dir):
            makedirs(self.inprogress_dir)


    def in_progress(self):
        with open(self.inprogress_dir + self.id + ".txt", "w") as f:
            f.writelines([
                "Start: UTC " + str(datetime.now(timezone.utc)),
                "\nGeneration: " + str(self.complete_gens + 1),
                "\nSLURM JOB ID: " + str(self.slurm_id),
            ])


    def not_in_progress(self) -> None:
        remove(self.inprogress_dir + self.id + ".txt")


    def detect_early_stop(self) -> bool:
        if not isinstance(self.tpot.early_stop, int):
            return False
        if len(self.gen_scores) < self.tpot.early_stop + 1:
            return False
        if len(set(str(s) for s in self.gen_scores[-(self.tpot.early_stop + 1):])) > 1:
            return False
        return True


    def export_fitted_pipeline(self) -> None:
        with open(self.export_dir + self.id + ".pkl", "wb") as f:
            dill.dump(self.tpot.fitted_pipeline_, f)


    def evaluate(self) -> None:
        # training_score = self.tpot.score(self.dataset.X, self.dataset.y)
        scores = {}
        ratings = {}
        for eval_random_state in self.eval_random_states:
            kfold_predictions, kfold_r2 = self.tpot_test(eval_random_state)
            scores[eval_random_state] = kfold_r2
            ratings[eval_random_state] = kfold_predictions.tolist()
        pipeline_parameters = {key: value for key, value in self.__dict__.items() if key in PIPELINE_PARAM_KEYS}
        tpot_parameters = {key: value for key, value in self.tpot.__dict__.items() if key in TPOT_PARAM_KEYS}
        tpot_parameters["search_space"] = self.config_search_space
        pipeline_attributes = {"complete_gens": self.complete_gens, "kfold_scores": scores, "kfold_ratings": ratings}
        tpot_attributes = {
            "fitted_pipeline_": str(self.tpot.fitted_pipeline_),
            "evaluated_individuals": str(self.tpot.evaluated_individuals),
            "pareto_front": str(self.tpot.pareto_front),
        }
        pipeline_dict = {
            "pipeline_parameters": pipeline_parameters,
            "tpot_parameters": tpot_parameters,
            "pipeline_attributes": pipeline_attributes,
            "tpot_attributes": tpot_attributes,
        }
        with open(self.output_dir + str(self.id) + ".json", "w") as f:
            json.dump(pipeline_dict, f, indent=4, default=TPOTPipeline.dict_everything)


    def tpot_test(self, eval_random_state: int) -> tuple[np.ndarray, float]:
        #set_param_recursive(self.tpot.fitted_pipeline_.steps, 'random_state', eval_random_state)
        kfold = KFold(n_splits=self.tpot.cv, shuffle=True, random_state=eval_random_state)
        kfold_predictions = kfold_predict(self.tpot.fitted_pipeline_, kfold, self.dataset)
        kfold_r2 = float(r2_score(kfold_predictions, self.dataset.y))
        return kfold_predictions, kfold_r2


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

