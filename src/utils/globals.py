from pathlib import Path
import os

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path("data")
RAW_PATH = Path(DATA_PATH / "01_raw")
PREPROCESSED_PATH = Path(DATA_PATH / "02_preprocessed")
FEATURE_PATH = Path(DATA_PATH / "03_features")
DROPLET_PATH = Path(DATA_PATH / "04_droplet")
RESULT_PATH = Path(DATA_PATH / "05_results")
EXPERIMENT_PATH = Path(PROJECT_PATH / "experiments")
