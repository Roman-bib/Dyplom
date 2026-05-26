"""
NeuralProphet — нейросетевая декомпозиция временного ряда (глава 2.6.3 ВКР).

Отличия от классического Prophet:
  - AR-Net: авторегрессионный компонент (n_lags периодов истории)
  - PyTorch backend вместо Stan → быстрее, GPU-поддержка
  - Точность на коротких горизонтах выше за счёт AR-Net
  - Современный API, активная разработка

Архитектура модели:
  Trend(changepoints) + Seasonality(Fourier) + AR-Net(n_lags → Dense → yhat)

Ключевые параметры:
  n_lags     — глубина авторегрессии (в периодах); автовычисляется как ~24ч
  n_forecasts=1 — one-step-ahead прогноз (согласован с XGBoost/LSTM)
  epochs     — 50 по умолчанию (достаточно для ВКР, не тормозит)

Сериализация:
  NeuralProphet.save() / NeuralProphet.load() — штатный способ (.np модель)
  Не используем joblib — модель содержит PyTorch state dict, joblib не умеет.

КРИТИЧНО по валидации:
  train_df и val_df ДОЛЖНЫ быть строго разными срезами.
  При вызове m.fit(train_df, validation_df=val_df) NeuralProphet обучается
  только на train_df и оценивает на val_df — честная схема.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Утилиты (дублируем из prophet_model — чтобы не создавать кросс-зависимость)
# ---------------------------------------------------------------------------

def _infer_freq(ds: pd.Series) -> str:
    """Определяет pandas-частоту из ряда дат."""
    ds = pd.to_datetime(ds).sort_values()
    if len(ds) < 3:
        return "h"
    inferred = pd.infer_freq(ds.iloc[:50])
    if inferred:
        return inferred
    deltas_sec = ds.diff().dropna().dt.total_seconds()
    step_sec = float(deltas_sec.median())
    if step_sec <= 90:
        return "1min"
    if step_sec <= 600:
        return f"{int(round(step_sec / 60))}min"
    if step_sec <= 3600 * 1.5:
        return "h"
    if step_sec <= 86400 * 1.5:
        return "D"
    return "h"


def _enough_for_yearly(ds: pd.Series) -> bool:
    span = (pd.to_datetime(ds.max()) - pd.to_datetime(ds.min())).days
    return span >= 540


def _auto_n_lags(ds: pd.Series) -> int:
    """
    Возвращает число лагов для AR-Net равное ~24 часам в периодах данных.
    Минимум 12, максимум 336 (чтобы не тормозило на очень мелком шаге).
    """
    ds = pd.to_datetime(ds).sort_values()
    if len(ds) < 2:
        return 24
    step_sec = float(ds.diff().dropna().dt.total_seconds().median())
    step_min = step_sec / 60
    n = int(round(24 * 60 / step_min))
    return max(12, min(n, 336))


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train_neural_prophet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_lags: Optional[int] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> object:
    """
    Обучает NeuralProphet с AR-Net компонентом.

    КРИТИЧНО: train_df и val_df — строго разные временные срезы.
    Обучение идёт ТОЛЬКО на train_df, оценка — на val_df.

    Parameters
    ----------
    train_df, val_df : DataFrame с колонками 'ds', 'y'
    n_lags           : окно AR-Net в периодах (None = автоматически ~24ч)
    epochs           : число эпох обучения
    save_path        : путь для сохранения модели (.np файл)

    Returns
    -------
    Обученная модель NeuralProphet
    """
    try:
        from neuralprophet import NeuralProphet
    except ImportError as exc:
        raise ImportError(
            "NeuralProphet не установлен. Установите:\n"
            "  pip install neuralprophet"
        ) from exc

    if "ds" not in train_df.columns or "y" not in train_df.columns:
        raise ValueError("train_df должен содержать колонки 'ds' и 'y'")

    train_df = train_df[["ds", "y"]].copy()
    val_df   = val_df[["ds", "y"]].copy()
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    val_df["ds"]   = pd.to_datetime(val_df["ds"])

    freq = _infer_freq(train_df["ds"])
    if n_lags is None:
        n_lags = _auto_n_lags(train_df["ds"])

    yearly = _enough_for_yearly(train_df["ds"])

    if verbose:
        print(f"  NeuralProphet: freq={freq}, n_lags={n_lags}, "
              f"epochs={epochs}, yearly={yearly}")

    # Подавляем многословный вывод NeuralProphet
    import logging
    logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
    logging.getLogger("NP.forecaster").setLevel(logging.ERROR)

    try:
        m = NeuralProphet(
            n_forecasts=1,
            n_lags=n_lags,
            yearly_seasonality=yearly,
            weekly_seasonality=True,
            daily_seasonality=True,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            trainer_config={"enable_progress_bar": False},
        )
    except TypeError:
        # Старые версии NeuralProphet не имеют trainer_config
        m = NeuralProphet(
            n_forecasts=1,
            n_lags=n_lags,
            yearly_seasonality=yearly,
            weekly_seasonality=True,
            daily_seasonality=True,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

    m.fit(train_df, freq=freq, validation_df=val_df)

    if verbose:
        print(f"  NeuralProphet обучен: n_lags={m.n_lags}, freq={freq}")

    if save_path:
        save_neural_prophet(m, save_path)

    return m


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def predict_neural_prophet(
    model,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """
    One-step-ahead прогноз на тестовом периоде.

    NeuralProphet с AR-Net нуждается в n_lags предшествующих значениях y
    для каждой предсказываемой точки. Передаём контекст из конца train+val.

    Это «teacher forcing» — в качестве AR-входа используются реальные
    (а не предсказанные) значения y. Это согласовано с подходом XGBoost,
    который тоже использует реальные лаговые значения через FeatureBuilder.

    Parameters
    ----------
    model        : обученная модель NeuralProphet
    train_val_df : объединённый train+val (контекст для AR-Net)
    test_df      : тестовый датасет (ds + y — y нужны как AR-вход)

    Returns
    -------
    np.ndarray длины len(test_df): прогноз RPS ≥ 0
    """
    n_lags = getattr(model, "n_lags", 0)

    test_df = test_df[["ds", "y"]].copy()
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    if n_lags > 0:
        context = train_val_df[["ds", "y"]].tail(n_lags).copy()
        context["ds"] = pd.to_datetime(context["ds"])
        df_pred = pd.concat([context, test_df], ignore_index=True)
    else:
        df_pred = test_df.copy()

    forecast = model.predict(df_pred)

    # Колонка прогноза называется 'yhat1' для n_forecasts=1
    if "yhat1" not in forecast.columns:
        # Fallback: ищем первую yhat-колонку
        yhat_cols = [c for c in forecast.columns if c.startswith("yhat")]
        if not yhat_cols:
            raise RuntimeError(
                f"NeuralProphet не вернул колонку yhat. "
                f"Доступные: {list(forecast.columns)}"
            )
        yhat_col = yhat_cols[0]
    else:
        yhat_col = "yhat1"

    preds = forecast[yhat_col].iloc[-len(test_df):].values
    return np.clip(np.asarray(preds, dtype=float), 0, None)


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------

def save_neural_prophet(model, path: str) -> None:
    """Сохраняет NeuralProphet через штатный метод (.np файл)."""
    np_path = path if path.endswith(".np") else path + ".np"
    os.makedirs(os.path.dirname(np_path) or ".", exist_ok=True)
    model.save(np_path)


def load_neural_prophet(path: str):
    """Загружает NeuralProphet из .np файла."""
    try:
        from neuralprophet import NeuralProphet
    except ImportError as exc:
        raise ImportError("pip install neuralprophet") from exc
    np_path = path if path.endswith(".np") else path + ".np"
    return NeuralProphet.load(np_path)
