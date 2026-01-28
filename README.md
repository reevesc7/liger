# liger

Likert General Regressor

A set of utilites for training and evaluating estimators on semantic analysis tasks.

## Introduction

This repository houses an experiment workflow created to analyze the feasibility
of using simple estimators to approximate subsets of LLM behaviors.

The liger package itself consists of utilities making up the core components of this workflow.
The module can be applied across many projects looking to accomplish any combination of
sentence embedding, LLM output analysis, (segmented) hyperparameter tuning, and
data analysis and visualization thereof.

This repository also includes several scripts and configurations in the `experiments/`
directory which interface with liger to complete the workflow used in the experiments
documented in \<TBD\>.
These scripts are not intended to be general-purpose utilities.

## Extras

liger has optional dependencies specified by multiple extras.
Extras are required to use certain features:

- `embedding`: Enables sentence embedding via SentenceTransformers and OpenAI
- `surveying`: Enables querying OpenAI models for prompt responses
- `tpot`: Enables running TPOT fitting with run segmentation and detailed outputs
- `all`: Includes all extras above

## Install only the liger package

To install the liger source code, install from the GitHub repository.
Use the following argument in a chosen package manager:

```
"liger[<any desired extras>] @ git+https://github.com/reevesc7/liger.git@main"
```

For example,

```
pip install "liger[embedding,surveying] @ git+https://github.com/reevesc7/liger.git@main"
```

```
uv add "liger[all] @ git+https://github.com/reevesc7/liger.git@main"
```

## Install liger as a project with experiment scripts

To work within a liger project environment, first clone the GitHub repository:

```
git clone https://github.com/reevesc7/liger.git
```

Then, in the project directory, install liger as editable with a chosen package manager.
For example:

```
pip install -e .[all]
```

```
uv sync --extra all
```

