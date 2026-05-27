"""
Система сравнения моделей прогнозирования (глава 2 ВКР).

Реализует единый интерфейс для обучения и оценки моделей на одних и тех
же данных и подбора лучшей по совокупности метрик.

Состав моделей (после аудита, см. AUDIT_REPORT.md, шаг 4):
  1. XGBoost   — основная (по ВКР)
  2. Prophet   — модель декомпозиции (тренд + сезонность)
  3. LSTM      — нейросетевой подход (multivariate, sliding window)

Linear Regression / Ridge удалена: см. models/linear_model.py.
В качестве time-series-baseline используется seasonal-naive,
встроенный отдельным методом сравнения.

Использование:
    comparator = ModelComparison()
    results = comparator.run(train, val, test)
    best_name, best_model = comparator.best_model()
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from preprocessing.feature_engineering import FeatureBuilder
from models.forecasters import (
    train_xgboost, train_xgboost_random_search,
    predict_xgboost,
    train_lstm_random_search,
)
from evaluation.metrics import (
    evaluate, print_comparison_table,
    peak_detection_metrics, peak_focused_mae,
)


class ModelComparison:
    """
    Запускает сравнительный эксперимент: одни данные, разные модели.

    Атрибуты после .run():
        results_       : dict {model_name -> metrics_dict}
        models_        : dict {model_name -> обученный объект модели}
        predictions_   : dict {model_name -> np.ndarray прогнозов на test}
        winner_        : имя лучшей модели по primary_metric
    """

    def __init__(
        self,
        model_save_dir: str = "./saved_models",
        primary_metric: str = "MAE",
        peak_quantile: float = 0.95,
    ):
        self.model_save_dir = model_save_dir
        self.primary_metric = primary_metric
        self.peak_quantile = peak_quantile
        self.results_: Dict[str, dict] = {}
        self.models_: Dict[str, object] = {}
        self.predictions_: Dict[str, np.ndarray] = {}
        self.winner_: Optional[str] = None
        try:
            import config as _cfg
            _exog = getattr(_cfg, "EXOG_COLS", [])
        except ImportError:
            _exog = []
        self._builder = FeatureBuilder(exog_cols=_exog)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def run(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        include_prophet: bool = True,
        include_lstm: bool = True,
        lstm_window_size: int = 24,
        lstm_epochs: int = 80,
        force_retrain: bool = False,
    ) -> Dict[str, dict]:
        """
        Обучает все модели и оценивает на test.

        Parameters
        ----------
        train, val, test : DataFrame с колонками ds, y
        include_prophet  : включить Prophet (медленно)
        include_lstm     : включить LSTM (требует TensorFlow)
        lstm_window_size : размер окна для LSTM в периодах
        force_retrain    : True — переобучить даже если сохранённые файлы есть
        """
        print("=" * 60)
        print("СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
        print("=" * 60)

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self._builder.transform_splits(train, val, test)

        # Порог пика — 95-й перцентиль фактического трафика на train+val
        peak_threshold = float(
            np.percentile(np.concatenate([y_train.values, y_val.values]),
                          self.peak_quantile * 100)
        )

        # --- 1. XGBoost (основная модель) ---
        xgb_path = os.path.join(self.model_save_dir, "xgboost.pkl")
        if not force_retrain and os.path.exists(xgb_path):
            print("\n[1] XGBoost — загружаем сохранённую модель...")
            model = joblib.load(xgb_path)
            preds = predict_xgboost(model, X_test)
            mae_train = float(np.mean(np.abs(y_train.values - predict_xgboost(model, X_train))))
            mae_val   = float(np.mean(np.abs(y_val.values   - predict_xgboost(model, X_val))))
            self._record("XGBoost", y_test, preds, peak_threshold, 0.0,
                         model_obj=model,
                         y_train_for_mase=np.concatenate([y_train.values, y_val.values]),
                         mae_train=mae_train, mae_val=mae_val)
        else:
            self._fit_xgboost(X_train, y_train, X_val, y_val,
                              X_test, y_test, peak_threshold)

        # --- 2. Prophet (опционально) ---
        prophet_path = os.path.join(self.model_save_dir, "prophet.np")
        if include_prophet:
            if not force_retrain and os.path.exists(prophet_path):
                print("\n[2] Prophet — загружаем сохранённую модель...")
                try:
                    from models.forecasters import load_prophet, predict_prophet
                    final_model = load_prophet(
                        os.path.join(self.model_save_dir, "prophet.json")
                    )
                    import config as _cfg
                    _exog = [c for c in getattr(_cfg, "EXOG_COLS", [])
                             if c in test.columns]
                    future_reg = test[["ds"] + _exog].copy() if _exog else None
                    prophet_pred_df = predict_prophet(
                        final_model, periods=len(test),
                        history_ds=train["ds"], future_regressors=future_reg,
                    )
                    preds = np.clip(prophet_pred_df["yhat"].values[:len(y_test)], 0, None)
                    self._record("Prophet", y_test, preds, peak_threshold, 0.0,
                                 model_obj=final_model,
                                 y_train_for_mase=np.concatenate([train["y"].values, val["y"].values]))
                except Exception as e:
                    print(f"  Загрузка Prophet не удалась ({e}), переобучаем...")
                    self._fit_prophet(train, val, test, y_test, peak_threshold)
            else:
                self._fit_prophet(train, val, test, y_test, peak_threshold)

        # --- 3. LSTM (опционально) ---
        lstm_path = os.path.join(self.model_save_dir, "lstm.keras")
        if include_lstm:
            if not force_retrain and os.path.exists(lstm_path):
                print("\n[3] LSTM — загружаем сохранённую модель...")
                try:
                    from models.forecasters import LSTMArtifact, predict_lstm_aligned
                    artifact = LSTMArtifact.load(os.path.join(self.model_save_dir, "lstm"))
                    X_full = pd.concat([X_train, X_val, X_test])
                    preds = predict_lstm_aligned(artifact, X_full=X_full, n_test=len(y_test))
                    self._record("LSTM", y_test, preds, peak_threshold, 0.0,
                                 model_obj=artifact,
                                 y_train_for_mase=np.concatenate([y_train.values, y_val.values]))
                except Exception as e:
                    print(f"  Загрузка LSTM не удалась ({e}), переобучаем...")
                    self._fit_lstm(train, val, test, X_train, y_train, X_val, y_val,
                                   X_test, y_test, window_size=lstm_window_size,
                                   epochs=lstm_epochs, peak_threshold=peak_threshold)
            else:
                self._fit_lstm(
                    train, val, test,
                    X_train, y_train, X_val, y_val,
                    X_test, y_test,
                    window_size=lstm_window_size,
                    epochs=lstm_epochs,
                    peak_threshold=peak_threshold,
                )

        # --- Итоговая таблица и выбор победителя ---
        self.winner_ = self._select_winner()
        print_comparison_table(self.results_)
        print(f"\nЛучшая модель по {self.primary_metric}: {self.winner_}")
        print(f"(Порог пика для peak-метрик: {peak_threshold:.1f} RPS)")
        print("=" * 60)

        return self.results_

    def best_model(self) -> Tuple[str, object]:
        if self.winner_ is None:
            raise RuntimeError("Вызовите .run() перед .best_model()")
        return self.winner_, self.models_[self.winner_]

    def save_best(self, path: Optional[str] = None) -> str:
        name, model = self.best_model()
        if path is None:
            os.makedirs(self.model_save_dir, exist_ok=True)
            path = os.path.join(self.model_save_dir, f"best_{name}.pkl")
        if name == "LSTM":
            # LSTM имеет свой формат сохранения
            from models.forecasters import LSTMArtifact
            lstm_path = os.path.join(self.model_save_dir, "best_LSTM")
            model.save(lstm_path)
            print(f"Best model (LSTM) saved -> {lstm_path}.keras + .meta.pkl")
            return lstm_path
        if name == "Prophet":
            from models.forecasters import save_prophet
            prophet_path = os.path.join(self.model_save_dir, "best_Prophet.json")
            save_prophet(model, prophet_path)
            print(f"Best model (Prophet) saved -> {prophet_path}")
            return prophet_path
        joblib.dump(model, path)
        print(f"Best model ({name}) saved -> {path}")
        return path

    # ------------------------------------------------------------------
    # Внутренние методы обучения
    # ------------------------------------------------------------------

    def _record(self, name: str, y_true, preds, peak_threshold: float,
                elapsed: float, model_obj=None, y_train_for_mase=None,
                mae_train: Optional[float] = None, mae_val: Optional[float] = None):
        """Считает все метрики и сохраняет в self.results_/models_/predictions_."""
        metrics = evaluate(y_true.values if hasattr(y_true, "values") else y_true,
                           preds, name, verbose=True,
                           y_train=y_train_for_mase)
        peak_m = peak_detection_metrics(
            y_true.values if hasattr(y_true, "values") else y_true,
            preds, peak_threshold,
        )
        metrics.update(peak_m)
        metrics["peak_focused_mae"] = peak_focused_mae(
            y_true.values if hasattr(y_true, "values") else y_true,
            preds, peak_threshold,
        )
        metrics["train_time_s"] = round(elapsed, 2)

        # --- Диагностика переобучения / недообучения ---
        if mae_train is not None and mae_val is not None:
            metrics["mae_train"] = round(mae_train, 4)
            metrics["mae_val"] = round(mae_val, 4)
            overfit_ratio = (mae_val - mae_train) / (mae_train + 1e-9)
            metrics["overfit_ratio"] = round(overfit_ratio, 4)

            if overfit_ratio > 0.30:
                print(f"  [!] ПЕРЕОБУЧЕНИЕ ({name}): MAE_val/MAE_train = "
                      f"{mae_val:.2f}/{mae_train:.2f} "
                      f"(разрыв {overfit_ratio*100:.1f}% > 30%)")
            elif mae_train > metrics.get("MAE", mae_val) * 0.85 and overfit_ratio < 0.05:
                print(f"  [!] НЕДООБУЧЕНИЕ ({name}): MAE_train={mae_train:.2f}, "
                      f"MAE_val={mae_val:.2f} — оба высоки, разрыв мал")
            else:
                print(f"  [OK] Баланс ({name}): MAE_train={mae_train:.2f}, "
                      f"MAE_val={mae_val:.2f} (разрыв {overfit_ratio*100:.1f}%)")

        self.results_[name] = metrics
        if model_obj is not None:
            self.models_[name] = model_obj
        self.predictions_[name] = np.asarray(preds, dtype=float)

    def _fit_xgboost(self, X_train, y_train, X_val, y_val, X_test, y_test, peak_threshold):
        print("\n[1] XGBoost (Random Search)...")
        t0 = time.time()
        save_path = os.path.join(self.model_save_dir, "xgboost.pkl")
        model, best_params, _ = train_xgboost_random_search(
            X_train, y_train, X_val, y_val, save_path=save_path
        )
        print(f"  Лучшие параметры: {best_params}")
        preds = predict_xgboost(model, X_test)
        elapsed = time.time() - t0

        y_tr_arr = np.asarray(y_train, dtype=float).flatten()
        y_vl_arr = np.asarray(y_val,   dtype=float).flatten()
        mae_train = float(np.mean(np.abs(y_tr_arr - predict_xgboost(model, X_train))))
        mae_val   = float(np.mean(np.abs(y_vl_arr - predict_xgboost(model, X_val))))

        self._record("XGBoost", y_test, preds, peak_threshold, elapsed,
                     model_obj=model,
                     y_train_for_mase=np.concatenate([y_tr_arr, y_vl_arr]),
                     mae_train=mae_train, mae_val=mae_val)

    def _fit_prophet(self, train, val, test, y_test, peak_threshold):
        print("\n[2] Prophet (grid-search по гиперпараметрам, train-only)...")
        try:
            from models.forecasters import (
                train_prophet, predict_prophet, refit_prophet_full, save_prophet
            )
            t0 = time.time()

            # КРИТИЧНО: train_df = ТОЛЬКО train, val_df = ТОЛЬКО val.
            # Иначе grid-search оценивает «обучен на train+val, проверен на val»
            # = утечка валидации в обучение.
            import config as _cfg
            _use_holidays = getattr(_cfg, "PROPHET_USE_HOLIDAYS", False)
            _country = getattr(_cfg, "PROPHET_COUNTRY_CODE", "RU")
            _exog = [c for c in getattr(_cfg, "EXOG_COLS", [])
                     if c in train.columns]

            # NeuralProphet получает ds+y без RobustSTL, но с заполнением пропусков
            # (интерполяция — шаг пайплайна, общий для всех моделей)
            def _fill_y(df):
                d = df[["ds", "y"] + [c for c in df.columns if c not in ("ds","y")]].copy()
                # method="linear" не требует DatetimeIndex (в отличие от "time")
                d["y"] = d["y"].interpolate(method="linear").bfill().ffill()
                # бинарные экзогенные флаги заполняем 0 (нет события = нет флага)
                for col in d.columns:
                    if col not in ("ds", "y") and d[col].isna().any():
                        d[col] = d[col].fillna(0)
                return d.reset_index(drop=True)

            best_model = train_prophet(
                train_df=_fill_y(train),
                val_df=_fill_y(val),
                save_path=None,        # сохраним позже после refit
                use_holidays=_use_holidays,
                country_code=_country,
                verbose=True,
                exog_cols=_exog,
            )
            # После grid-search дообучаем на train+val для финального предсказания
            final_model = refit_prophet_full(
                base_model=best_model,
                train_val_df=_fill_y(pd.concat([train, val]).sort_values("ds").reset_index(drop=True)),
                use_holidays=_use_holidays,
                country_code=_country,
                exog_cols=_exog,
            )

            # Будущие значения экзогенных переменных берём из test (известны заранее)
            future_reg = test[["ds"] + _exog].copy() if _exog else None

            # Авто-определение freq из train.ds
            prophet_pred_df = predict_prophet(
                final_model,
                periods=len(test),
                history_ds=train["ds"],
                future_regressors=future_reg,
            )
            preds = prophet_pred_df["yhat"].values[:len(y_test)]
            preds = np.clip(preds, 0, None)

            save_path = os.path.join(self.model_save_dir, "prophet.json")
            save_prophet(final_model, save_path)

            elapsed = time.time() - t0
            # y_train_for_mase должен быть без NaN — берём заполненные версии,
            # иначе sklearn-метрики (MASE) падают с "Input contains NaN"
            _tv_filled = pd.concat(
                [_fill_y(train), _fill_y(val)]
            ).sort_values("ds").reset_index(drop=True)
            self._record("Prophet", y_test, preds, peak_threshold, elapsed,
                         model_obj=final_model,
                         y_train_for_mase=_tv_filled["y"].values)
        except Exception as e:
            print(f"  Prophet пропущен: {e}")

    def _fit_lstm(
        self, train, val, test,
        X_train, y_train, X_val, y_val, X_test, y_test,
        window_size: int, epochs: int, peak_threshold: float,
    ):
        print(f"\n[3] LSTM (Random Search, window_size={window_size}, "
              f"Huber loss, fit-on-train scaler)...")
        try:
            from models.forecasters import predict_lstm_aligned, LSTMArtifact
            t0 = time.time()
            save_path = os.path.join(self.model_save_dir, "lstm")
            artifact = train_lstm_random_search(
                X_train, y_train, X_val, y_val,
                window_size=window_size,
                final_epochs=epochs,
                save_path=save_path,
            )

            # Context-aware inference: окна формируются на полной матрице
            # train+val+test признаков, чтобы первые n_test точек получили
            # полный контекст из train+val.
            X_full = pd.concat([X_train, X_val, X_test])
            preds = predict_lstm_aligned(
                artifact, X_full=X_full, n_test=len(y_test),
            )

            elapsed = time.time() - t0

            # Train/val MAE из истории обучения Keras
            mae_train_lstm = mae_val_lstm = None
            hist = getattr(artifact, "history_", None)
            if hist and "mae" in hist and "val_mae" in hist:
                mae_train_lstm = float(hist["mae"][-1])
                mae_val_lstm   = float(hist["val_mae"][-1])
            elif hist and "loss" in hist and "val_loss" in hist:
                mae_train_lstm = float(hist["loss"][-1])
                mae_val_lstm   = float(hist["val_loss"][-1])

            self._record("LSTM", y_test, preds, peak_threshold, elapsed,
                         model_obj=artifact,
                         y_train_for_mase=np.concatenate([
                             np.asarray(y_train, dtype=float).flatten(),
                             np.asarray(y_val,   dtype=float).flatten(),
                         ]),
                         mae_train=mae_train_lstm, mae_val=mae_val_lstm)
        except ImportError as e:
            print(f"  LSTM пропущен (нет TensorFlow): {e}")
        except Exception as e:
            print(f"  LSTM пропущен: {e}")

    def _select_winner(self) -> str:
        """Выбирает победителя по primary_metric."""
        candidates = self.results_
        if not candidates:
            return min(self.results_,
                       key=lambda n: self.results_[n].get(self.primary_metric, float("inf")))
        return min(
            candidates,
            key=lambda name: candidates[name].get(self.primary_metric, float("inf")),
        )
