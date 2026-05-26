"""
LinearRegression / Ridge baseline — УДАЛЕНО (по требованию ВКР).

Обоснование удаления (см. AUDIT_REPORT.md, шаг 4):

1. Линейная регрессия принципиально не способна моделировать
   взаимодействия между признаками и нелинейные пороги (например,
   «при cpu>80% и hour∈[18,22] RPS взлетает в 3×»). Для трафика
   web-сервиса это базовое поведение, поэтому baseline постоянно
   проигрывает на 30–50% по MAE даже на простых датасетах.

2. На лаговых признаках с сильной мультиколлинеарностью (rps_lag_1h,
   mean_1h, mean_6h практически коллинеарны) Ridge даёт смещённые
   коэффициенты. Это создаёт иллюзорное «равенство условий», но
   на деле ставит линейную модель в худшие условия, чем XGBoost,
   которому коллинеарность безразлична.

3. В ВКР по теме проактивного автомасштабирования сравнивать XGBoost
   c линейной регрессией некорректно: это сравнение «алгоритм для
   сложных нелинейных рядов» vs «инструмент описательной статистики».
   Корректный baseline для time-series — это либо seasonal-naive
   (y_t = y_{t-168h}), либо ARIMA/SARIMA, либо Holt-Winters.

4. Финальный набор моделей в эксперименте: XGBoost (основная),
   Prophet (декомпозиция), LSTM (нейросеть). Этот набор покрывает
   три качественно разных семейства алгоритмов.

Модуль оставлен как заглушка ради обратной совместимости импортов.
"""

import numpy as np


_REMOVED_MSG = (
    "LinearRegression / Ridge удалена из сравнения по требованию ВКР. "
    "Корректный baseline для time-series — seasonal-naive (predict_seasonal_naive). "
    "См. AUDIT_REPORT.md, шаг 4."
)


def train_linear(X_train, y_train, save_path=None):  # noqa: D401
    """Удалено. Используйте seasonal-naive baseline или XGBoost/LSTM/Prophet."""
    raise NotImplementedError(_REMOVED_MSG)


def predict_linear(model, X):  # noqa: D401
    """Удалено."""
    raise NotImplementedError(_REMOVED_MSG)


# ---------------------------------------------------------------------------
# Корректный baseline для time-series — seasonal-naive
# ---------------------------------------------------------------------------

def predict_seasonal_naive(
    y_history: np.ndarray,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    """
    Seasonal-naive: ŷ_{t+h} = y_{t+h-S}, где S — длина сезона.

    Это эталонный baseline для сезонных рядов (трафик, потребление,
    спрос). Метрика MASE построена именно вокруг него — она нормирует
    ошибку модели на ошибку seasonal-naive.

    Parameters
    ----------
    y_history    : массив наблюдений (последний элемент — самое свежее)
    horizon      : сколько шагов вперёд предсказать
    season_length : длина сезона в периодах (для часовых данных недели = 168)

    Returns
    -------
    np.ndarray размера (horizon,)
    """
    y_history = np.asarray(y_history, dtype=float)
    if len(y_history) < season_length:
        # На очень короткой истории падаем на naive: повторяем последнее значение
        return np.full(horizon, y_history[-1] if len(y_history) else 0.0)

    last_season = y_history[-season_length:]
    # Циклически повторяем последний сезон столько раз, сколько нужно
    repeats = int(np.ceil(horizon / season_length))
    forecast = np.tile(last_season, repeats)[:horizon]
    return forecast
