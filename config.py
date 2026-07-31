# config.py

DATASETS = {
    "bank_marketing": {"openml_id": 44126, "task": "classification"},
    "california": {"openml_id": 44025, "task": "regression"},
    "magic_telescope": {"openml_id": 44125, "task": "classification"},
}

MASTER_SEED = 42
MAX_SAMPLES = 10000
TEST_SIZE = 0.3

OPENML_CACHE_DIR = "openml_cache"  # project-local fetch_openml cache; add to .gitignore

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