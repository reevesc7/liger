# Liger-Testbed

A directory with scripts for formatting datasets for use with `liger`.

Here are scripts for
- parsing Smallville outputs into prompts,
- embedding prompts,
- surveying models for responses to prompts,
- deriving functionals from model probability distribution responses,
- and synthesizing all relevant fields into an aggregate dataset.

**Required Packages:** `liger`

## Setting up environment

### 0. Requirements

You will need a working
[Anaconda](https://www.anaconda.com/download),
[Miniconda](https://docs.anaconda.com/miniconda/),
or [Miniforge](https://github.com/conda-forge/miniforge)
installation.

### 1. Create a Conda environment to work in:

```
conda create -n liger python=3.10
```

### 2. Activate Conda environment:

```
conda activate liger
```

### 3. Install requirements:

While in the directory of the desired experiment,

```
pip install -r requirements.txt
```

### 4. :


