#  NeuralSurv: Deep Survival Analysis with Bayesian Uncertainty Quantification

Monod, Micheli & Bhatt (2025). NeuralSurv: Deep Survival Analysis with Bayesian Uncertainty Quantification. arXiv. [DOI](
https://doi.org/10.48550/arXiv.2505.11054)

## Warranty
Imperial makes no representation or warranty about the accuracy or completeness of the data nor that the results will not constitute in infringement of third-party rights. Imperial accepts no liability or responsibility for any use which may be made of any results, for the results, nor for any reliance which may be placed on any such work or results.

## Cite

```bibtex
@misc{monod2025neuralsurvdeepsurvivalanalysis,
      title={NeuralSurv: Deep Survival Analysis with Bayesian Uncertainty Quantification}, 
      author={Mélodie Monod and Alessandro Micheli and Samir Bhatt},
      year={2025},
      eprint={2505.11054},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.11054}, 
}
```

## System Requirements
* macOS or UNIX
* This release has been checked on Ubuntu 22.04.4 LTS and macOS Sonoma 14.1.2

## Installation
Clone the repository. A `yml` file is provided and can be used to build a conda virtual environment containing all dependencies. Create the environment using:

```bash
cd neuralsurv
conda env create -f neuralsurv.yml
```


## Usage 

The file `template.py` demonstrates all the steps shown below, providing a complete workflow from data preparation to model fitting, posterior sampling, visualization, and evaluation.

### Prepare data
The NeuralSurv framework expects:

- `time_train` and `time_test`: Event or censoring times as JAX arrays.
- `event_train` and `event_test`: Event indicators (1 if event occurred, 0 if censored) as JAX arrays.
- `x_train` and `x_test`: Covariate features as JAX arrays.

Example using synthetic data:
```python
key = jr.PRNGKey(12)
n_train, n_test, p = 10, 5, 3

time_train = jax.random.normal(jr.split(key)[0], (n_train,)) * 100 + 150
event_train = jax.random.bernoulli(jr.split(key)[1], 0.5, (n_train,)).astype(jnp.int32)
x_train = jax.random.normal(jr.split(key)[2], (n_train, p))

time_test = jax.random.normal(jr.split(key)[3], (n_test,)) * 100 + 150
event_test = jax.random.bernoulli(jr.split(key)[4], 0.5, (n_test,)).astype(jnp.int32)
x_test = jax.random.normal(jr.split(key)[5], (n_test, p))
```

Don't forget to rescale your time so the start time is 0.
```python
min_time_train = time_train.min()
time_train = time_train - min_time_train
time_test = time_test - min_time_train
```

### Specify Model and Training Parameters

#### Priors

Set prior parameters for the Bayesian model:
```python
alpha_prior = 1.0
beta_prior = 1.0
rho = jnp.float32(1.0)
```

####  Algorithm Parameters

Control EM and CAVI optimization iterations:

```python
max_iter_em = 200 # maximum iteration for the EM algorithm
max_iter_cavi = 200 # maximum iteration for the CAVI algorithm
num_points_integral_em = 1000 # Number of points in trapezoidal approx
num_points_integral_cavi = 1000 # Number of points in trapezoidal approx
```

####  Training Parameters
```python
batch_size = 1000
```

####  Neural Network Architecture
```python
n_hidden = 2
n_layers = 2
activation = jax.nn.relu
```

#### Posterior Sampling

Number of posterior samples to draw:
```python
num_samples = 1000
```

### Load environments and NeuralSurv class
```python

import jax
import jax.numpy as jnp
import jax.random as jr

from sksurv.nonparametric import kaplan_meier_estimator

from model.model import NMLP, MLP
from neuralsurv import NeuralSurv
from postprocessing.plot import (
    plot_posterior_phi,
    plot_hazard_function,
    plot_survival_function_vs_km_curve,
)

```


### Fit the Model

#### Create model instance:

```python
model = NMLP(mlp_main=MLP(n_hidden=n_hidden, n_layers=n_layers, activation=activation))
```

#### Initialize parameters:

```python
key, step_rng = jr.split(key)
model_params_init = model.init(step_rng, jnp.array([0]), jnp.zeros(p))
```

#### Instantiate NeuralSurv:

```python
neuralsurv = NeuralSurv.load_or_create(
    model,
    model_params_init,
    alpha_prior,
    beta_prior,
    rho,
    num_points_integral_em,
    num_points_integral_cavi,
    batch_size,
    max_iter_em,
    max_iter_cavi,
    output_dir,
)
```

#### Fit model to training data:

```python
neuralsurv.fit(time_train, event_train, x_train)
```

### Posterior and Prior Sampling

After fitting the model, you can draw posterior and prior samples:

```python
key, step_rng = jr.split(key)
neuralsurv.get_posterior_samples(step_rng, num_samples)

key, step_rng = jr.split(key)
neuralsurv.get_prior_samples(step_rng, num_samples)
```

###  Plotting Functions

#### Posterior Distribution of $\phi$
```python
plot_posterior_phi(
    neuralsurv.posterior_params["alpha"],
    neuralsurv.posterior_params["beta"],
    plot_dir,
)
```

#### Hazard Function
```python
times = jnp.linspace(1e-6, time_train.max(), num=100)
plot_hazard_function(neuralsurv.predict_hazard_function(times, x_train, aggregate=(0,2)), plot_dir, "train")
```

#### Survival Function vs Kaplan-Meier
```python
from sksurv.nonparametric import kaplan_meier_estimator

km_times_train, km_survival_train = kaplan_meier_estimator(event_train.astype(bool), time_train)
surv_train = neuralsurv.predict_survival_function(times, x_train, aggregate=(0,2))
plot_survival_function_vs_km_curve(surv_train, km_times_train, km_survival_train, plot_dir, "train")
```

###  Compute Evaluation Metrics

You can compute concordance index (c-index), Brier score, D-calibration and KM calibration:

```python
neuralsurv.compute_evaluation_metrics(time_train, event_train, time_test, event_test, x_test, plot_dir)
```



## Reproduce results of the paper


</details>

<details>
<summary> Reproduce the benchmark results </summary>

### 1. Setup

In `main_benchmark.py`, specify the directory where the results will be stored (`output_dir`). For example,
```python
output_dir = "/Users/melodiemonod/projects/2025/neuralsurv/benchmark"
```

### 2. Running Experiments
Run `main_benchmark.py`.

</details>

<details>
<summary> Reproduce results of experiment "Synthetic Data Experiment" </summary>

### 1. Setup
First, specify the following entries in `config_experiment_1.json`

* Dataset Directory (`dataset_dir`): The directory where the repository is located + '`data/data_files`'. 
* GPU name (`devices`) and index (`devices_index`): The name and index of your GPU device.

For example,
```json
 "dataset_dir":"/home/mm3218/git/neuralsurv/data/data_files",
 "devices": ["NVIDIA RTX A6000"],
 "devices_index":"0"
```

Second, specify the following directories at the top of the `submit_job_experiment_1.sh` file:

* Repository Directory (`INDIR`): The directory where the repository is located.
* Output Directory (`OUTDIR`): The directory where the results will be stored.

```bash
INDIR="/home/mm3218/git/neuralsurv"
OUTDIR="/home/mm3218/projects/2025/neuralsurv"
```

Third, open a terminal and navigate to the repository directory, then execute the `submit_job_experiment_1.sh` script:

```bash
cd neuralsurv
bash submit_job_experiment_1.sh
```

The script will generate folders in the output directory, one for each experiment.


### 2. Running Experiments

Go to the output directory, locate the experiment folder and navigate into it. 
```bash
cd $OUTDIR
cd $DATE-synthetic_25
```

Run NeuralSurv and obtain the evaluation metrics and predict the survival function: 
```bash
bash $DATE-synthetic_25.sh
```

Repeat these steps for each experiment folder created in `$OUTDIR`. 

### 3. Reproduce table and figure
To reproduce the figure and the table, run `make_tables_figures/synthetic_figure.py` and `make_tables_figures/synthetic_table.py` by specifying the correct `date`, `dataset_name`, `jobid` and `jobid_neuralsurv`. 

</details>

<details>
<summary> Reproduce results of experiment "Real Survival Data Experiments" </summary>

### 1. Setup
First, specify the following entries in `config_experiment_2.json`

* Dataset Directory (`dataset_dir`): The directory where the repository is located + '`data/data_files`'. 
* GPU name (`devices`) and index (`devices_index): The name and index of your GPU device.

For example,
```json
 "dataset_dir":"/home/mm3218/git/neuralsurv/data/data_files",
 "devices": ["NVIDIA RTX A6000"],
 "devices_index":"0"
```

Second, specify the following directories at the top of the `submit_job_experiment_2.sh` file:

* Repository Directory (`INDIR`): The directory where the repository is located.
* Output Directory (`OUTDIR`): The directory where the results will be stored.

```bash
INDIR="/home/mm3218/git/neuralsurv"
OUTDIR="/home/mm3218/projects/2025/neuralsurv"
```

Third, open a terminal and navigate to the repository directory, then execute the `submit_job_experiment_2.sh` script:

```bash
cd neuralsurv
bash submit_job_experiment_2.sh
```

The script will generate folders in the output directory, one for each experiment.


### 2. Running experiments

Go to the output directory, locate the experiment folder and navigate into it. 
```bash
cd $OUTDIR
cd $DATE-colon_sub_125_fold_0_layers_2_hidden_16_relu
```

Run NeuralSurv and obtain the evaluation metrics and predict the survival function: 
```bash
bash $DATE-colon_sub_125_fold_0_layers_2_hidden_16_relu.sh
```

Repeat these steps for each fold of a dataset created in `$OUTDIR`. 

### 3. Reproduce table 

To reproduce the table, run `make_tables_figures/real_data_tables.py` by specifying the correct `date`, `dataset_name`, `jobid` and `base` and `suffix` of the  `jobid_neuralsurv`. 



