# Partial Replication: Why Do Tree-Based Models Outperform Deep Learning on Tabular Data?

A lightweight replication of selected findings from:

> Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).
> *Why do tree-based models still outperform deep learning on tabular data?*
> arXiv:2207.08815

---

## Overview

This project partially replicates the core benchmark and two empirical findings
from Grinsztajn et al. (2022) using 3 tabular datasets, standard scikit-learn
models, and CPU-only training.

The original paper benchmarks tree-based models against neural networks across
45 datasets and investigates *why* tree models often outperform deep learning
on medium-sized tabular data. This replication focuses on three specific claims:

1. Tree-based models outperform MLPs on many tabular datasets
2. MLPs degrade more severely under uninformative (noise) features
3. Tree-based models are sensitive to feature-space rotations, while MLPs are not

**Scope note:** The original work evaluates PyTorch-based MLP, ResNet,
FT-Transformer, and SAINT architectures with extensive GPU-based
hyperparameter searches over 45 datasets. This project uses sklearn's
`MLPClassifier` / `MLPRegressor` on 3 datasets with fixed hyperparameters.

The results are therefore intended as a **qualitative replication of selected
findings**, not a full reproduction of the original benchmark.

---

## Datasets

All datasets are drawn from the paper's OpenML benchmark suite.
Datasets are capped at 10,000 samples to match the paper's
medium-scale tabular setting.

| Dataset            | Task           | Samples | Features | OpenML ID |
| ------------------ | -------------- | ------- | -------- | --------- |
| Bank Marketing     | Classification | 10,000  | 7        | 44126     |
| California Housing | Regression     | 10,000  | 8        | 44025     |
| Magic Telescope    | Classification | 10,000  | 10       | 44125     |

---

## Models

### Tree-based Models

* Random Forest
* Gradient Boosting Trees (GBT)
* XGBoost

### Neural Network

* sklearn `MLPClassifier` / `MLPRegressor`
* 2 hidden layers of 256 units
* Adam optimizer with early stopping

---

## Preprocessing Protocol

The original paper applies different preprocessing pipelines to tree models
and neural networks. This replication follows the same design:

* **Tree models** receive raw input features
* **MLP** receives Gaussianized features using
  `QuantileTransformer(output_distribution="normal")`

This distinction is important because tree-based methods are largely
scale-invariant, while neural networks are sensitive to feature scaling
and heterogeneous marginal distributions.

Additional preprocessing:

* Missing rows removed
* Classification labels encoded
* 70/30 train-test split
* Stratified splitting for classification datasets

---

# Experiments

## 1. Benchmark (`benchmark.py`)

Direct comparison of all four models on all three datasets.

### Results

| Dataset                    | RF    | GBT   | XGBoost | MLP   |
| -------------------------- | ----- | ----- | ------- | ----- |
| Bank Marketing (Accuracy)  | 0.800 | 0.803 | 0.789   | 0.774 |
| California Housing (R²)    | 0.653 | 0.668 | 0.675   | 0.622 |
| Magic Telescope (Accuracy) | 0.858 | 0.854 | 0.859   | 0.855 |

Tree-based models outperform MLP on Bank Marketing and California Housing.
On Magic Telescope the gap is very small (≤ 0.004), consistent with the
paper's observation that the advantage magnitude varies substantially
across datasets.

![Benchmark](fig1_benchmark.png)

---

## 2. Sensitivity to Uninformative Features (`finding2.py`)

Random Gaussian noise features (0, 5, 10, 20, 50) are appended to each dataset.
Performance is measured for GBT and MLP under increasing feature noise.

Results are averaged across 5 random seeds.

### California Housing (R²)

| Noise Features | GBT           | MLP           |
| -------------- | ------------- | ------------- |
| 0              | 0.657 ± 0.000 | 0.622 ± 0.001 |
| 50             | 0.630 ± 0.005 | 0.362 ± 0.014 |

GBT drops by only 0.027, while MLP drops by 0.260.

A similar pattern appears across all three datasets:
tree-based models remain comparatively robust as irrelevant features increase,
while MLP performance deteriorates substantially.

This behavior qualitatively aligns with Finding 2 of the original paper.

![Finding 2](fig2_finding2.png)

---

## 3. Rotation Invariance (`finding3.py`)

Features are first Gaussianized, then random orthogonal rotations are applied
using `scipy.stats.special_ortho_group`.

Unlike the benchmark experiment, both GBT and MLP receive identical rotated
inputs in order to isolate the effect of feature-space orientation.

Results are averaged across:

* 10 random rotation matrices
* 3 model initialization seeds

### California Housing (R²)

| Setting  | GBT           | MLP           |
| -------- | ------------- | ------------- |
| Original | 0.657 ± 0.000 | 0.622 ± 0.002 |
| Rotated  | 0.550 ± 0.009 | 0.624 ± 0.004 |

GBT performance drops significantly after rotation, while MLP remains nearly
unchanged.

On Magic Telescope, MLP slightly surpasses GBT after rotation,
matching the qualitative trend reported in the original paper.

These results support the hypothesis that tree-based methods exploit
axis-aligned structure in tabular datasets, whereas MLPs are approximately
rotation-invariant.

![Finding 3](fig3_finding3.png)

---

# How to Run

### Requirements

* Python 3.9+
* CPU-only execution
* Approximate runtime: 20–30 minutes total

Install dependencies:

```bash
pip install scikit-learn xgboost matplotlib seaborn scipy pandas numpy
```

Download the ARFF datasets from OpenML and place them in the project root:

* `bank_marketing.arff` — https://www.openml.org/d/44126
* `california.arff` — https://www.openml.org/d/44025
* `magic_telescope.arff` — https://www.openml.org/d/44125

Run experiments:

```bash
python benchmark.py
python finding2.py
python finding3.py
python visualize.py
```

Approximate runtimes:

| Script         | Runtime |
| -------------- | ------- |
| `benchmark.py` | ~2 min  |
| `finding2.py`  | ~8 min  |
| `finding3.py`  | ~15 min |
| `visualize.py` | <1 min  |

Generated outputs:

* CSV result files
* Publication-style PNG figures

---

# Project Structure

```text
├── data_loader.py
├── benchmark.py
├── finding2.py
├── finding3.py
├── visualize.py
├── benchmark_results.csv
├── finding2_results.csv
├── finding3_results.csv
├── fig1_benchmark.png
├── fig2_finding2.png
└── fig3_finding3.png
```

---

# Limitations

* Only 3 datasets are evaluated (vs. 45 in the original paper)
* sklearn MLP is used instead of the paper's PyTorch implementation
* No large-scale hyperparameter search is performed
* Transformer-based tabular architectures are not included
* Finding 1 from the original paper is not replicated
* No formal statistical significance testing is conducted

Accordingly, the results should be interpreted as a lightweight qualitative
replication rather than a definitive benchmark study.

---

# Reference

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).
*Why do tree-based models still outperform deep learning on tabular data?*
arXiv:2207.08815

Original benchmark code:
https://github.com/LeoGrin/tabular-benchmark

