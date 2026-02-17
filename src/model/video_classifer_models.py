from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[[], BaseEstimator]
    task: str                 # "classification" | "regression"
    family: str               # "tree", "boosting"
    default_params: Dict[str, object]

class Classifier:
    _REGISTRY: Dict[str, ModelSpec]={
        "xgboost": ModelSpec(
            builder= lambda: XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ),
        task="classification",
        family="boosting",
        default_params={
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6
        }
        ),
        "random_forest": ModelSpec(
            builder=lambda: RandomForestClassifier(random_state=0),
            task="classification",
            family="tree",
            default_params={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2
            }
        ),
        "logistic_regression": ModelSpec(
            builder=lambda: LogisticRegression(
                solver='saga',
                max_iter=1000
            ),
            task="classification",
            family="linear_model",
            default_params={
                "C":1
            }
        )
    }


    def __init__(self, model_name: str):
        self.model_type = model_name
        if self.model_type not in self._REGISTRY:
            supported = ", ".join(sorted(self._REGISTRY.keys()))
            raise ValueError(
                f"Unsupported model type: '{model_name}'. Supported: {supported}"
            )
    
    def load(self)-> BaseEstimator:
        model_spec=self._REGISTRY[self.model_type]
        return model_spec.builder
    
    def model_spec(self)-> ModelSpec:
        return self._REGISTRY[self.model_type]