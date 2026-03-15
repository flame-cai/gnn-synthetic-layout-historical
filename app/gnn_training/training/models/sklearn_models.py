# training/models/sklearn_models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

def get_sklearn_model(model_name: str, config: dict) -> BaseEstimator:
    """Factory function to create a scikit-learn model."""
    model_cfg = config['model_configs'][model_name]
    
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=model_cfg['n_estimators'],
            max_depth=model_cfg['max_depth'],
            random_state=config['random_seed'],
            n_jobs=model_cfg['n_jobs']
        )
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=model_cfg['n_estimators'],
            learning_rate=model_cfg['learning_rate'],
            max_depth=model_cfg['max_depth'],
            random_state=config['random_seed']
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            C=model_cfg['C'],
            max_iter=model_cfg['max_iter'],
            random_state=config['random_seed'],
            n_jobs=model_cfg.get('n_jobs', -1)
        )
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")