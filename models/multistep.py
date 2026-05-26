"""
Многошаговое прогнозирование (multi-step forecasting).

Реализует рекурсивный прогноз на горизонте H > 1 шаг для моделей,
обученных на one-step ahead. Работает по схеме direct-recursive:
  1. Получить прогноз ŷ_{t+1} от модели
  2. Подставить ŷ_{t+1} как новое наблюдение в историю
  3. Пересчитать признаки X_{t+2} с этим псевдо-наблюдением
  4. Получить ŷ_{t+2}, повторить до t+H

Альтернатива production-систем (например, AWS Predictive Scaling),
которые делают прогноз на 48 часов вперёд.

Ограничения:
  • Накопление ошибки: ŷ_{t+h} использует все предыдущие
    предсказания как «факт», поэтому при h → ∞ дисперсия растёт.
  • Применимо только для одношаговых моделей XGBoost/LSTM;
    Prophet поддерживает многошаговый прогноз нативно.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from preprocessing.feature_engineering import FeatureBuilder


def recursive_forecast(
    history_df: pd.DataFrame,
    model: object,
    predict_fn: Callable,
    builder: FeatureBuilder,
    horizon: int,
    step_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """
    Рекурсивный многошаговый прогноз.

    Parameters
    ----------
    history_df    : DataFrame с колонками ds, y (вся доступная история)
    model         : обученная модель (XGBoost, LSTMArtifact, ...)
    predict_fn    : функция predict(model, X) -> np.ndarray
    builder       : FeatureBuilder для построения признаков
    horizon       : сколько шагов вперёд предсказать
    step_minutes  : шаг ряда; если None — определяется автоматически

    Returns
    -------
    DataFrame с колонками ds, y_hat — прогноз на horizon шагов вперёд.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = history_df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Определяем шаг ряда
    if step_minutes is None:
        deltas = df["ds"].diff().dropna().dt.total_seconds() / 60
        step_minutes = float(deltas.median()) if len(deltas) else 60.0

    step = pd.Timedelta(minutes=step_minutes)
    last_ts = df["ds"].iloc[-1]

    forecasts = []
    for h in range(1, horizon + 1):
        next_ts = last_ts + h * step

        # Строим признаки на расширенной истории
        # (включая псевдо-наблюдения для уже сделанных шагов)
        # Добавляем метку на t+h как NaN, чтобы FeatureBuilder построил
        # для неё лаги (на основе уже накопленных y и предыдущих ŷ).
        synthetic = df.copy()
        # Добавим строку с NaN — она нужна, чтобы признаки сместились
        # на нужный t+h. Но FeatureBuilder dropna удалит её, поэтому
        # подставим псевдо-y из последнего прогноза или среднего.
        if h > 1:
            # Используем последний прогноз как «факт» для построения лагов
            synthetic = pd.concat(
                [synthetic, pd.DataFrame({
                    "ds": [next_ts],
                    "y": [forecasts[-1]["y_hat"]],
                })],
                ignore_index=True,
            )

        # Берём последнюю строку признаков
        X = builder.get_X(synthetic)
        if len(X) == 0:
            forecasts.append({"ds": next_ts, "y_hat": np.nan})
            continue

        x_last = X.iloc[[-1]]
        y_pred = float(predict_fn(model, x_last)[0])
        y_pred = max(0.0, y_pred)
        forecasts.append({"ds": next_ts, "y_hat": y_pred})

    return pd.DataFrame(forecasts)
