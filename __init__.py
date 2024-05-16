from .dataset import Dataset
from .data_io import change_sentences, add_sentences, embed_dataframe, random_endpts_dataset, random_scores_dataset, plot2D, plot_expl_var_ratios
from .operations import interpolate_points, vector_projection, vector_rejection, compute_R2
from .pipelines import tpot_pipeline
from .training_testing import train_model, evaluate_model, kfold_predict


__all__ = [
    "Dataset",
    "change_sentences",
    "add_sentences",
    "embed_dataframe",
    "random_endpts_dataset",
    "random_scores_dataset",
    "plot2D",
    "plot_expl_var_ratios",
    "interpolate_points",
    "vector_projection",
    "vector_rejection",
    "compute_R2",
    "tpot_pipeline",
    "train_model",
    "evaluate_model",
    "kfold_predict"
]