# Liger-Testbed

A simple repository made to act as a working directory for the Likert General Regressor project.

For now, only test scripts are included in this repository.

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

### 4. Test installation:

With `liger` installed, you can run a test.

```
bash test_bash.sh
```

This should run a TPOT pipeline for a few generations.

Before running any generations, the script should create the
`in_progress/`,
`outputs/`,
and `slurmout/`
directories, and the `test_data/` directory in `outputs/`

Upon each generation's completion, the following should print:

```
PIPELINE ID: <run ID, string (UTC time at run's start)>
TPOT RANDOM STATE: 1
<Any warnings appear here>
Generation:  <generation number: int | float.0>
Best mean_squared_error score: <prediction score: float>
Best complexity_scorer score: <complexity score: float>
```

Each generation should create `pipeline_data.json` and `population.pkl` files
in the `outputs/test_data/<run ID>/` directory
(each generation will override the previous generation's file).

Upon completion of the last generation, the following should be the case:
- The script printed a summary of the fitted pipeline
- The script quit with the messages `RUN COMPLETE` and `ENDING RECURSION`
- A `fitted_pipeline.pkl` file representing the run's best performing TPOT model is in the `outputs/test_data/<run ID>/` directory
- A `pipeline_data.json` file is in the `Outputs/smallville_poignancy_avstd_llmembed/<run ID>/` directory
    - This JSON file should contain most of the run's information,
      including `"target_gens": 10` and `"complete_gens": <n complete gens>`,
      displaying that the run was set to complete 10 generations but may have carried out fewer.
      (this is because `"early_stop": 2` can terminate the run early,
      after no improvement across the last 3 generations)
    - You can see a record of each generation's score(s) in the `pipeline_data.json` file,
      under `"gen_scores"` and confirm that the last 3 generations had the same score for each measure.

## Slurm Configuration

Before running scripts via Slurm,
you will need to adjust a couple lines of code in the `slurm_runner.sb` file.

Every change is noted by a comment starting with `CHANGE`.
1. Change the the setting of the `PATH` variable to your `condabin` and `anaconda3/bin` directories
2. Change the setting of the `HOME` variable to your home directory (usually `/mnt/home/<username>`)
3. Change the `cd` command into the working directory to the location of your `Liger-Testbed` directory

Once you have adjusted the content of `slurm_runner.sb`,
you may run Slurm jobs with it by executing `test_slurm.sh` in the terminal.

```
bash test_slurm.sh
```

You can adjust the range of TPOT random state used for the jobs
(and therefore the number of jobs which will be submitted)
by adding the `-s (start)` and `-e (end)` (non-inclusive) variables in `test_slurm.sh`.
Using the `-n (n jobs)` argument when calling `slurm_submitter.sh` will defer to the
config file for determining random states.
The current file is set to submit a single TPOT pipeline with a TPOT random state
determined by the config file (`-n 1`).

To leave TPOT random states as random values, use the `-n (n jobs)` argument *instead* of `-s` and `-e` and use the `"random_state": null` in the config file.

