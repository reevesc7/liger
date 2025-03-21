# liger
Likert General Regressor

A set of helper functions for training and evaluating sklearn estimators on semantic analysis tasks.

## Installation

### Install Conda

[Instructions for installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Create a Conda environment

In a terminal window, run the following command:

```
conda create -n liger -y python=3.10
```

### Activate your Conda environment

In a terminal window, run the following command:

```
conda activate liger
```

### Install liger and dependencies

In a terminal window, run the following command:

```
pip install git+https://github.com/reevesc7/liger.git@main#egg=liger
```

## Editable Installations

For those wishing to contribute to `liger`, having an editable installation is desirable
in order to view the effects of code changes without needing to reinstall the package.

Because `liger` uses Poetry as a build backend,
simply running `pip install -e git+<repository URL>` will not properly install an editable version.
A Poetry installation is required to install an editable version.

### Install Poetry

[Instructions for installing Poetry](https://python-poetry.org/docs/)

### Create and activate a Conda environment

Like the normal installation, we will use a Conda environment.

In a terminal window, run the following commands:

```
conda create -n liger-edit -y python=3.10
conda activate liger-edit
```

*If you know Poetry and would rather use a Poetry virtual environment, you may
[create and activate a Poetry environment instead](https://python-poetry.org/docs/managing-environments/).*

### Clone this repository

In whichever directory you would like to store the `liger` source code, run the following command:

```
git clone https://github.com/reevesc7/liger.git
```

### Install dependencies

In the base directory of your local clone of `liger` (`cd liger`), run the following command:

```
poetry install
```

### Reinstall `liger` as editable version

Again, in the base directory of you local clone of `liger`, run the following command:

```
pip install -e .
```

Now any changes you make to the source code will immediately take effect when using
the `liger-edit` Conda environment.

