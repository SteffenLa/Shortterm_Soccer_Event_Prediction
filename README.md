# Lang et al. (in Review) Which indicators matter? Using performance indicators to predict in-game success-related events in association football. International Journal of Computer Science in Sport (IJCSS).

## Generic Experiment structure written in Python

### Setup

We are using [Conda](https://docs.conda.io/en/latest/miniconda.html) as dependency management system.
You can install the necessary libraries by performing

```
conda env create -f environment.yml
```

The created environment can be started by performing

```
conda activate goalpred
```

In case you already installed the environment and there is an update in the environment.yml file, perform

```
conda env update --file environment.yml --prune
```

### Structure of repository

The repository is structured as follows:

- configs folder: contains a training configuration file for the experiment
- data folder: contains an original match data file used in the experiments discribed in the study
- models folder: contains various ML model implementations that can be used in experiments
- output folder: contains the output files of the experiments and plots
- utils folder: contains various utility functions that can be used in experiments (data sampling, visualization, etc.)
- notebooks folder: contains notebooks that were created in the IDP, however, they are not necessary to run experiments
  or create plots (see next section)

### Automatic generation of plots

The `plot_creation.py` script can be used to automatically generate various plots for the available data.
The script can be run by performing `python plot_creation.py CONFIG` where `CONFIG` is the name of the configuration
file for the plots. The configuration file is located in the `configs` folder. Sample configuration files are already
available in the `configs` folder. A README file is also available in the `configs` folder.

### Execution of experiments

The `run.py` script can be used to run the experiments. The script can be run by performing `python run.py CONFIG` where
`CONFIG` is the name of the configuration file for the experiment. The configuration file has to be located in
the `configs` folder. A README file and sample configurations are available in the `configs` folder.

The `run.sbatch` script is a SLURM script that can be used on the CV group cluster. Could be modified for any other
cluster.