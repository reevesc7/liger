from typing import Any
import sys
from os import makedirs, remove
from os.path import isdir, isfile
import json
from random import randint
from datetime import datetime, timezone
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
TPOT_ATTRS = [
    "fitted_pipeline_",
    "pareto_front_fitted_pipelines_",
    "evaluated_individuals_",
]
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
    "tree_structure",
    "evaluated_individuals_",
    "fitted_pipeline_"
    "log_file_",
    "pareto_front_fitted_pipelines_",
}


class TPOTPipeline:

    def __init__(
        self,
        config_file: str,
        data_file: str,
        tpot_random_state: int | None = None,
        slurm_id: int | None = None,
        id: str | None = None,
        complete_gens: int | None = None,
    ) -> None:
        self.config_file = config_file
        self.data_file = data_file
        self.data_name = TPOTPipeline.get_filename(data_file)
        self.dataset = Dataset.from_csv(data_file)
        self.slurm_id = slurm_id
        if complete_gens is not None:
            self.complete_gens = complete_gens
        else:
            self.complete_gens = 0

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
        random_state = TPOTPipeline.use_first(
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

        # Create search space
        self.config_search_space = tpot_parameters["search_space"]
        search_space = create_search_space(self.config_search_space, random_state)

        # Set output paths
        self.output_dir = OUTPUT + self.data_name + "/"
        self.export_dir = EXPORT + self.data_name + "/"
        self.checkpoint_dir = CHECKPOINT + self.data_name + "/" + self.id + "/"
        self.inprogress_dir = IN_PROGRESS + self.data_name + "/"

        self.tpot = TPOTEstimator(
            search_space=search_space,
            periodic_checkpoint_folder=self.checkpoint_dir,
            random_state=random_state,
            **{key: value for key, value in tpot_parameters.items() if key not in [
                "search_space",
                "survival_selector",
                "parent_selector",
                "periodic_checkpoint_folder",
                "random_state",
            ]},
        )
        print("GENS", self.tpot.generations, flush=True)
        print("POPSIZE", self.tpot.population_size, flush=True)


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
    def find_checkpoint(data_name: str, id: str) -> str | None:
        checkpoint_name = CHECKPOINT + data_name + "/" + id + "/tpot_pipeline.json"
        if not isdir(CHECKPOINT + data_name + "/" + id):
            return None
        if isfile(checkpoint_name):
            return checkpoint_name
        return None


    @staticmethod
    def dict_everything(objec: Any) -> str | None:
        if isinstance(objec, np.ndarray):
            objec = objec.tolist()
            return json.dumps(objec, indent=4, default=TPOTPipeline.dict_everything)
        elif isinstance(objec, range):
            objec = [i for i in objec]
            return json.dumps(objec, indent=4, default=TPOTPipeline.dict_everything)
        elif hasattr(objec, "__dict__"):
            return json.dumps(objec.__dict__, indent=4, default=TPOTPipeline.dict_everything)
        else:
            return None


    def create_checkpoint(self) -> None:
        pipeline_data = self.get_pipeline_data()
        with open(self.checkpoint_dir + "tpot_pipeline.json", "w") as f:
            json.dump(pipeline_data, f)


    def get_pipeline_data(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key in [
            "config_file",
            "data_file",
            "tpot_random_state",
            "slurm_id",
            "id",
            "complete_gens",
        ]}


    def run_1_gen(self) -> None:
        # Capture output dynamically
        capture = LiveOutputCapture()
        sys.stdout = capture

        # Save prep
        self.save_prep()
        self.in_progress()
        print("\nRUNNING PIPELINE:", self.id, flush=True)
        print("TPOT RANDOM STATE:", self.tpot.random_state, flush=True)
        print("GENERATION:", self.complete_gens + 1, flush=True)
        if self.complete_gens < self.target_gens:
            self.tpot.fit(self.dataset.X, self.dataset.y)
            self.complete_gens += 1
        if self.complete_gens >= self.target_gens or "Will end the optimization process." in capture.get_output():
            self.export_fitted_pipeline()
            self.evaluate()
            self.create_checkpoint()
            self.not_in_progress()
            print("\nRUN COMPLETE")
            return
        self.create_checkpoint()
        self.not_in_progress()
        print(f"\nRUN INCOMPLETE WITH ID: {self.id}")
        return


    def save_prep(self) -> None:
        if not isdir(self.output_dir):
            makedirs(self.output_dir)
        if not isdir(self.export_dir):
            makedirs(self.export_dir)
        if not isdir(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)
        if not isdir(self.inprogress_dir):
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
        tpot_attributes = {key: value for key, value in self.tpot.__dict__.items() if key in TPOT_ATTR_KEYS}
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

