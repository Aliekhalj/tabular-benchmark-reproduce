# Partial Replication of *Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?*

A lightweight replication of selected experiments from:

> **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).**
> *Why do tree-based models still outperform deep learning on tabular data?*
> https://arxiv.org/abs/2207.08815

---

## Overview

This project reproduces three core experiments from the paper using a simplified CPU-only pipeline built with scikit-learn and XGBoost.

Unlike the original work, which evaluates many neural architectures across 45 datasets with extensive hyperparameter optimization, this project focuses on reproducing the qualitative behavior of the paper on three representative OpenML datasets.

The project includes:

- Benchmark comparison between tree models and MLP
- Effect of adding uninformative (noise) features
- Effect of random feature-space rotations
- Automatic dataset downloading through OpenML
- Publication-style visualizations

---

# Repository Structure

```
.
├── benchmark.py
├── config.py
├── data_loader.py
├── finding2.py
├── finding3.py
├── models.py
├── visualize.py
│
├── benchmark_results.csv
├── finding2_results.csv
├── finding3_results.csv
│
├── fig1_benchmark.png
├── fig2_finding2.png
├── fig3_finding3.png
│
├── openml_cache/
└── README.md
```

---

# Datasets

Datasets are downloaded automatically from OpenML using fixed `data_id`s.

No manual downloads are required.

| Dataset | Task | Samples | Features | OpenML ID |
|----------|------|---------|----------|-----------|
| Bank Marketing | Classification | 10,000 | 7 | 44126 |
| California Housing | Regression | 10,000 | 8 | 44025 |
| Magic Telescope | Classification | 10,000 | 10 | 44125 |

The first execution downloads the datasets and stores them in a local cache.
Subsequent runs use the cached copies.

---

# Models

Tree-based models:

- Random Forest
- Gradient Boosting Trees (GBT)
- XGBoost

Neural network:

- sklearn MLPClassifier
- sklearn MLPRegressor

The benchmark follows the preprocessing protocol of the original paper:

- Tree models receive raw features.
- MLP receives Gaussianized features using `QuantileTransformer(output_distribution="normal")`.

---

# Experiments

## 1. Benchmark

Compares four models on all datasets.

### Results

| Dataset | RF | GBT | XGBoost | MLP |
|----------|------|------|------|------|
| Bank Marketing (Accuracy) | **0.8003** | **0.8030** | 0.7887 | 0.7743 |
| California Housing (R²) | 0.8025 | 0.8198 | **0.8284** | 0.7643 |
| Magic Telescope (Accuracy) | 0.8577 | 0.8543 | **0.8587** | 0.8547 |

<p align="center">
<img src="fig1_benchmark.png" width="900">
</p>

---

## 2. Sensitivity to Uninformative Features

Random Gaussian noise features are appended to the datasets.

Performance is averaged over five random seeds.

### Summary

As the number of noise features increases:

- Tree models degrade slowly.
- MLP performance deteriorates substantially faster.

Example (California Housing):

| Noise Features | GBT (R²) | MLP (R²) |
|----------------|-----------|-----------|
| 0 | 0.8130 | 0.7638 |
| 5 | 0.8090 | 0.7147 |
| 10 | 0.8088 | 0.6856 |
| 20 | 0.8047 | 0.6348 |
| 50 | 0.8025 | 0.5019 |

<p align="center">
<img src="fig2_finding2.png" width="900">
</p>

---

## 3. Rotation Invariance

Following the paper, features are first Gaussianized and then rotated using random orthogonal matrices.

Unlike the benchmark experiment, **both** GBT and MLP receive identical rotated inputs.

Results are averaged across:

- 10 random rotations
- 3 model initialization seeds

Example (California Housing):

| Setting | GBT (R²) | MLP (R²) |
|----------|-----------|-----------|
| Original | 0.8129 | 0.7640 |
| Rotated | 0.7127 | 0.7647 |

Tree models lose significant performance after rotation, whereas the MLP is nearly rotation invariant.

<p align="center">
<img src="fig3_finding3.png" width="900">
</p>

---

# Running the Project

## Requirements

- Python 3.11+
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- scipy

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost
```

---

## Run Experiments

Benchmark:

```bash
python benchmark.py
```

Noise feature experiment:

```bash
python finding2.py
```

Rotation experiment:

```bash
python finding3.py
```

Generate figures:

```bash
python visualize.py
```

---

# Reproducibility

All datasets are referenced by immutable OpenML `data_id`s.

Classification datasets use stratified train/test splitting.

Random seeds are fixed through the configuration file to ensure reproducible experiments.

---

# Limitations

This project is intentionally a lightweight replication.

Compared to the original paper:

- only 3 datasets are evaluated (vs. 45)
- sklearn MLP is used instead of the PyTorch implementation
- no large-scale hyperparameter optimization is performed
- transformer-based tabular models are not included
- experiments are CPU-only
- results should be interpreted qualitatively rather than as an exact reproduction

---

# References

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).

**Why do tree-based models still outperform deep learning on tabular data?**

https://arxiv.org/abs/2207.08815

Original benchmark repository:

https://github.com/LeoGrin/tabular-benchmark