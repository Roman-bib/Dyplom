"""
XGBoost — основная прогностическая модель (глава 2.6.2 ВКР).

Особенности реализации:
  - Optuna TPE для подбора гиперпараметров (n_trials=25)
  - Early stopping по валидационной выборке на каждом trial
  - Квантильная регрессия для доверительных интервалов (q=0.1 и q=0.9)
  - Сохранение важности признаков
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ---------------------------------------------------------------------------
# Основная модель (точечный прогноз)
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 20,
    save_path: str = None,
    n_trials: int = 25,
):
    """
    Обучает XGBRegressor с Optuna TPE поиском гиперпараметров.

    Parameters
    ----------
    X_train, y_train      : обучающая выборка
    X_val,   y_val        : валидационная выборка (early stopping + Optuna)
    early_stopping_rounds : раундов без улучшения до остановки
    save_path             : путь для сохранения через joblib
    n_trials              : число Optuna-trials (25 ≈ хорошее качество за ~1 мин)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _use_optuna = True
    except ImportError:
        _use_optuna = False

    def _fit(params: dict):
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            verbosity=0,
            random_state=42,
            **params,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return m

    if _use_optuna:
        def objective(trial):
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            }
            m = _fit(params)
            preds = np.clip(m.predict(X_val), 0, None)
            return float(np.mean(np.abs(y_val - preds)))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        print(f"  XGBoost Optuna best: {best_params}  →  val MAE={study.best_value:.2f}")
        model = _fit(best_params)
    else:
        model = _fit(dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
        ))

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(model, save_path)

    print(f"  XGBoost обучен: {model.best_iteration + 1} деревьев "
          f"(early stopping @ {early_stopping_rounds})")
    return model


def predict_xgboost(model, X) -> np.ndarray:
    """Точечный прогноз. Значения < 0 обнуляются."""
    preds = model.predict(X)
    return np.clip(preds, 0, None)


# ---------------------------------------------------------------------------
# Квантильная регрессия — доверительные интервалы
# ---------------------------------------------------------------------------

def train_xgboost_quantile(
    X_train,
    y_train,
    X_val,
    y_val,
    quantile: float = 0.9,
    n_estimators: int = 300,
    early_stopping_rounds: int = 20,
    save_path: str = None,
):
    """
    Обучает модель квантильной регрессии XGBoost.
    Для 80%-го доверительного интервала нужны q=0.1 и q=0.9.
    """
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="quantile",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(model, save_path)

    return model


def get_confidence_interval(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    save_dir: str = None,
):
    """
    Возвращает нижнюю и верхнюю границы прогноза.

    Returns
    -------
    (lower, upper) — два numpy-массива того же размера, что X_test
    """
    if len(X_val) == 0 or len(X_test) == 0:
        n = len(X_test) if len(X_test) > 0 else len(X_train)
        return np.zeros(n), np.zeros(n)

    model_low = train_xgboost_quantile(
        X_train, y_train, X_val, y_val,
        quantile=lower_q,
        save_path=os.path.join(save_dir, "xgb_q10.pkl") if save_dir else None,
    )
    model_high = train_xgboost_quantile(
        X_train, y_train, X_val, y_val,
        quantile=upper_q,
        save_path=os.path.join(save_dir, "xgb_q90.pkl") if save_dir else None,
    )
    lower = np.clip(model_low.predict(X_test), 0, None)
    upper = np.clip(model_high.predict(X_test), 0, None)
    return lower, upper


# ---------------------------------------------------------------------------
# Важность признаков
# ---------------------------------------------------------------------------

def feature_importance(model, feature_names=None) -> pd.DataFrame:
    """
    Возвращает DataFrame с важностью признаков XGBoost (gain).
    """
    imp = model.feature_importances_
    names = feature_names if feature_names is not None else [
        f"f{i}" for i in range(len(imp))
    ]
    df = pd.DataFrame({"feature": names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
