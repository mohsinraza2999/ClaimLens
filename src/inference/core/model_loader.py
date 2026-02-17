from joblib import load
from pathlib import Path
from sklearn.base import BaseEstimator
def load_model()-> BaseEstimator:
    path=Path("artifacts", "models", "production_model.joblib")
    return load(path)