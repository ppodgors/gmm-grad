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

## Running Experiments

All experiments in this repository are managed through the `main.py` script. The general command structure is:

```bash
python main.py <experiment_name> [action] [options]
```

## Synthetic Data Experiment

This is the primary experiment of the study, designed to evaluate model performance on synthetic datasets across a range of feature dimensionalities (`d`).

The core logic for this experiment is located in the `src/experiments/synthetic/` directory.

---

### Running the Experiment

The experiment is managed via the `main.py` script using the `synthetic` command, which has two actions: `run` and `plot`.

**Running the Simulation**

To run the main simulation, use the `run` action. You must specify the number of samples per component. The **default** parameters are configured to reproduce the **exact results presented in our publication**.

```bash
python main.py synthetic run --n_samples 50
```

**Plotting the Results**

After the simulation is complete, you can generate plots from the saved results file using the `plot` action.

```bash
python main.py synthetic plot --input_file simulation_results/results_n50.npz
```
### Command-Line Arguments
| Argument          | Default | Description                                               |
|-------------------|---------|-----------------------------------------------------------|
| `--n_samples`     |         | **(Required)** The number of samples per component.       |
| `--num_worlds`    | `25`    | The number of different synthetic "worlds" to generate.   |
| `--num_reps`      | `4`     | The number of replications for each world.                |
| `--d_start`       | `20`    | The starting dimensionality of the feature space.         |
| `--d_stop`        | `101`   | The exclusive ending dimensionality of the feature space. |
| `--d_step`        | `10`    | The step size for the dimensionality range.               |

### Output

After running the `main.py synthetic run` command, the script saves the numerical results to the `simulation_results/` directory.

* **Format**: The results are stored in a NumPy (`.npz`) archive file.
* **Filename**: The name is based on the number of samples (`n`), e.g., `results_n100.npz`.

Each `.npz` file contains the following data arrays:
* `d_values`: The range of feature dimensions tested.
* `n_value`: The number of samples per component used in the run.
* `ari_grad`, `nmi_grad`, `time_grad`: Metrics for the GMM-grad method.
* `ari_gs`, `nmi_gs`, `time_gs`: Metrics for the GMM-GS method.
* `ari_gmm`, `nmi_gmm`, `time_gmm`: Metrics for the standard GMM.

## Olivetti Faces Experiment
This experiment evaluates clustering performance on the Olivetti Faces dataset, with the goal of automatically grouping photos of the same individual.
The core logic for this experiment is located in the `src/experiments/olivetti/` directory.

---

### Running the Experiment
he experiment is managed via the `main.py` script using the `olivetti` command.

The **default parameters** of this script are configured to reproduce the **exact results presented in our publication**. To run the experiment:

```bash
python main.py --experiment olivetti
```

### Command-Line Arguments

| Argument       | Default            | Description                                          |
|----------------|--------------------|------------------------------------------------------|
| `--n_rep`      | `25`               | The number of experiment repetitions for averaging.  |
| `--main_seed`  | `27`               | The main random seed for initialization.             |
| `--subjects`    | `10`               | The number of subjects (classes) from the dataset.   |
| `--output_dir` | `Olivetti_results` | The directory where the output file will be saved.   |

### Output

After execution, the script will produce:

1.  A summary of the final mean **ARI** and **NMI** scores, averaged over all repetitions and printed to the console.
2.  A detailed results file named `results.pkl`, saved in the specified output directory (`Olivetti_results/` by default). This file is a Python pickle object containing a dictionary with the raw metrics from each repetition.

## 20 Newsgroups Experiment

This experiment evaluates the clustering performance of regularized and standard GMMs on vectorized text data from the 20 Newsgroups dataset. The process involves creating document embeddings using a TF-IDF weighted average of `word2vec-google-news-300` word vectors.

The core logic for this experiment is located in the `src/experiments/newsgroups/` directory.

---

**Note**: The first time you run the experiment, the script will automatically download the `NLTK` stopwords corpus if it is not found on your system. The `word2vec-google-news-300` model (1.6 GB) will also be downloaded, which may take some time.


### Running the Experiment
This experiment uses an automated workflow. The `main` script checks if the necessary datasets exist; if not, it generates them before running the clustering analysis.

The **default** parameters are configured to reproduce the **exact results presented in our publication**. To run the experiment with the default configuration:

```bash
python main.py newsgroups
```

The first run will be significantly longer as it includes the one-time data generation step. Subsequent runs will be much faster as they will reuse the existing data files.

### Command-Line Arguments

| Argument            | Default                                               | Description                                                     |
|---------------------|-------------------------------------------------------|-----------------------------------------------------------------|
| `--worlds`          | `15`                                                  | The number of data variations ("worlds") to generate and test.  |
| `--reps`            | `10`                                                  | The number of repetitions with different seeds per world.       |
| `--samples_per_cat` | `100`                                                 | The number of documents to sample per category.                 |
| `--data_dir`        | `words`                                               | The directory to store the generated datasets and results.      |
| `--categories`      | `rec.sport.baseball soc.religion.christian sci.space` | A space-separated list of newsgroup categories to include.      |

### Output

After execution, the script will produce:

1.  A summary of the final mean **ARI** and **NMI** scores, averaged over all worlds and repetitions, printed to the console.
2.  A detailed results file saved in the specified data directory (`newsgroups_results/` by default). The filename indicates the number of worlds and repetitions (e.g., `results_w15_r10.pkl`). This file is a Python pickle object containing a dictionary with the raw metrics from every run.