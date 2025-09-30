# Adaptive Regularization in the Expectation–Maximization Algorithm for Gaussian Mixture Models

This repository contains the source code and a series of experiments for research on regularized EM algorithm for Gaussian Mixture Models (GMMs).
The primary goal is to compare the performance of a standard GMM against two custom regularized GMMs. 


---

## Project Overview

The core of this project is to compare three GMM-based clustering methods:
1.  **Standard GMM**: The baseline implementation from `scikit-learn`.
2.  **GMM-GS (Grid Search)**: A regularized GMM where the regularization hyperparameter ($\eta$) is tuned via a cross-validated grid search.
3.  **GMM-Grad (Gradient)**: A regularized GMM where the same hyperparameter is tuned using a gradient-based optimization method.

The performance of these models is evaluated using the **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** metrics across various data dimensionalities (`d`) and sample sizes (`n`).

---

## Setup

The project requires the following Python libraries:
* `numpy`
* `scikit-learn`
* `torch`
* `matplotlib`

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```
## Running the Experiment

The main experiment script is `run_synthetic_experiment.py`. It is configured via command-line arguments.

### Command-Line Arguments
* `--n_samples`: (Required) The number of samples per component/cluster.
* `--num_worlds`: (Optional) The number of different synthetic "worlds" (sets of true parameters) to generate. Default: `25`.
* `--num_reps`: (Optional) The number of random datasets (replications) to generate for each world. Default: `4`.
* `--d_start`: (Optional) The starting dimensionality of the feature space. Default: `20`.
* `--d_stop`: (Optional) The exclusive ending dimensionality of the feature space. Default: `101`.
* `--d_step`: (Optional) The step size for the dimensionality range. Default: `10`.

### Example Usage
To run the full experiment for `n=50` samples per component, with 25 worlds and 4 replications each:

```bash
python main.py --n_samples 50
```
## Results

After running `run_synthetic_experiment.py`, the numerical results are saved in the `simulation_results/` directory, which is created automatically.

* **Format**: The results are stored in a NumPy `.npz` archive file. 
* **Filename Convention**: The files are named based on the number of samples (`n`) used in the experiment.
    * Example: `simulation_results/results_n50.npz`

Each `.npz` file contains the following arrays:
* `d_values`: The range of dimensions tested.
* `n_value`: The number of samples per component.
* `ari_grad`, `nmi_grad`: ARI and NMI scores for the GMM-grad method.
* `ari_gs`, `nmi_gs`: ARI and NMI scores for the GMM-GS method.
* `ari_gmm`, `nmi_gmm`: ARI and NMI scores for the standard GMM.

### Generating Plots

To visualize the results, use the `plot_results_synthetic_exp.py` script. 

**Usage:**

You must provide the path to the input results file using the `--input_file` argument.

```bash
python plot_results_synthetic_exp.py --input_file simulation_results/results_n50.npz
