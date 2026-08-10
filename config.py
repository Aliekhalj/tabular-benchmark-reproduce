DATASETS = {
    "bank_marketing": {"openml_id": 44126, "task": "classification"},
    "california": {"openml_id": 44025, "task": "regression"},
    "magic_telescope": {"openml_id": 44125, "task": "classification"},
    "electricity": {"openml_id": 44120, "task": "classification"},
    "house_16H": {"openml_id": 44123, "task": "classification"},
    "credit": {"openml_id": 44089, "task": "classification"},
    "phoneme": {"openml_id": 44127, "task": "classification"},
    "wine": {"openml_id": 44091, "task": "classification"},
    "cpu_act": {"openml_id": 44132, "task": "regression"},
    "elevators": {"openml_id": 44134, "task": "regression"},
    "wine_quality": {"openml_id": 44136, "task": "regression"},
    "diamonds": {"openml_id": 44140, "task": "regression"},
    "isolet": {"openml_id": 44135, "task": "regression"},
}

MASTER_SEED = 42
MAX_SAMPLES = 10000
TEST_SIZE = 0.3

OPENML_CACHE_DIR = "openml_cache"

MLP_HIDDEN_LAYER_SIZES = (256, 256)
MLP_MAX_ITER = 1000
MLP_VALIDATION_FRACTION = 0.1
MLP_N_ITER_NO_CHANGE = 20

BENCHMARK_N_ESTIMATORS = 300
FINDING_N_ESTIMATORS = 200

NOISE_LEVELS = [0, 5, 10, 20, 50]
NOISE_SEEDS = [42, 7, 13, 21, 99]

N_ROTATIONS = 10
ROTATION_MODEL_SEEDS = [42, 7, 13]

# Phase 2 Commit 2: tuning
TUNING_N_ITER = 50
# 9% of total matches the paper's own val:total ratio (70/9/21 train/val/test);
# expressed as a fraction of OUR 70%-train split: 0.09 / 0.70.
TUNING_VAL_FRACTION = 9 / 70
# Arbitrary, chosen only to be clearly distinct from MASTER_SEED/NOISE_SEEDS/
# ROTATION_MODEL_SEEDS -- the paper doesn't specify tuning-search seeds.
TUNING_VAL_SPLIT_SEED = 501
TUNING_SEARCH_SEED = 502