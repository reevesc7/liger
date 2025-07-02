# Liger-Testbed

**Required Packages:** `liger`

A directory with scripts for formatting datasets for use with `liger`.

Here are scripts for
- parsing Smallville outputs into prompts,
- embedding prompts,
- surveying models for responses to prompts,
- deriving functionals from model probability distribution responses,
- and synthesizing all relevant fields into an aggregate dataset.

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

### 4. Copy `example_config.py`:

The file `config.py` is ignored by git,
allowing users to modify it as needed for experiments.

```
cp example_config.py config.py
```

### 5. Modify configuration as needed:

The behavior of the scripts in this directory are controlled by `config.py`.
See below for an outline of how to modify the config.

## Formatting data

### Modifying configuration

### Isolating formatting steps

Each segment of the dataset formatting process can be run separately.
Simply run, for instance, `python get_prompts.py` to create a prompts file.

All config parameters will be obeyed.
E.g., `PROMPTS_OP = Op.RETR` will still look to retrieve prompts from
the `PROMPTS_FILE` filepath.
There are two exceptions:
- `Op.SKIP` behaves like `Op.RETR`
- An output file is always written
  at the filepath that would also be used to retrieve that data,
  even if `SAVE_INTERMEDIATES = False` or a file already exists at that location

