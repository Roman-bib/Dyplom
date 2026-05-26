"""
Метрик-агностичный FeatureBuilder для прогнозирования временных рядов.

Поддерживает любые целевые метрики:
  rps, concurrent_users, cpu_usage, memory_usage, latency_ms и пр.

Группы признаков:
  1. Лаговые          — y_lag_1h, y_lag_24h, y_lag_168h
  2. Скользящие       — y_mean/std/min/max за 1ч, 6ч, 24ч
  3. Производные      — y_diff (1-я разность) + y_pct_change (отн. изменение)
  4. Календарные      — hour, day_of_week, is_weekend, hour_sin/cos, dow_sin/cos
  5. Нормирующие      — y_zscore_24h, y_ratio_to_max_24h (scale-free)
  6. Spike-индикаторы — y_is_spike_prev, y_spike_count_24h, y_periods_since_spike
                        (сигнал кластеризации пиков → шанс не «сглаживать» их)

Адаптивность:
  • Лаги, выходящие за длину истории, автоматически исключаются.
  • Все признаки вычисляются БЕЗ утечки таргета: rolling/diff используют
    shift(1), лаги — shift(n).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

try:
    import holidays as _holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False


logger = logging.getLogger(__name__)


def _infer_step_minutes(index: pd.DatetimeIndex) -> float:
    """Шаг временного ряда в минутах (медиана разностей)."""
    if len(index) < 2:
        return 60.0
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 60
    return float(deltas.median())


def _hours_to_periods(hours: float, step_minutes: float) -> int:
    """Конвертация часов в число периодов (строк) при заданном шаге."""
    periods = hours * 60 / step_minutes
    return max(1, int(round(periods)))


class FeatureBuilder:
    """
    Конструктор признаков. Стандартные параметры подходят для любого
    регулярного временного ряда; адаптируется к доступной истории.
    """

    LAG_HOURS = [1, 24, 168]
    ROLL_HOURS = [1, 6, 24]
    MIN_HISTORY_MULTIPLIER = 2.0

    def __init__(
        self,
        lag_hours: Optional[List[float]] = None,
        roll_hours: Optional[List[float]] = None,
        include_calendar_cyclic: bool = True,
        include_pct_change: bool = True,
        include_zscore: bool = True,
        include_min_max: bool = True,
        include_spike_features: bool = True,
        spike_sigma: float = 2.0,
        include_holiday_features: bool = True,
        country_code: str = "RU",
        exog_cols: Optional[List[str]] = None,
    ):
        self.lag_hours = lag_hours if lag_hours is not None else self.LAG_HOURS
        self.roll_hours = roll_hours if roll_hours is not None else self.ROLL_HOURS
        self.include_calendar_cyclic = include_calendar_cyclic
        self.include_pct_change = include_pct_change
        self.include_zscore = include_zscore
        self.include_min_max = include_min_max
        self.include_spike_features = include_spike_features
        self.spike_sigma = float(spike_sigma)
        self.include_holiday_features = include_holiday_features and _HOLIDAYS_AVAILABLE
        self.country_code = country_code
        self.exog_cols = exog_cols or []

        self.last_built_features_: List[str] = []
        self.last_step_min_: Optional[float] = None
        self.last_n_rows_: Optional[int] = None
        self.last_skipped_lags_: List[float] = []
        self._warned_exog: set = set()

    # ------------------------------------------------------------------
    # Основной API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ds" not in df.columns or "y" not in df.columns:
            raise ValueError("DataFrame должен содержать колонки 'ds' и 'y'")

        data = df.copy()
        data["ds"] = pd.to_datetime(data["ds"])
        data = data.set_index("ds").sort_index()

        n = len(data)
        step_min = _infer_step_minutes(data.index)
        self.last_step_min_ = step_min
        self.last_n_rows_ = n

        active_lags, skipped = self._select_active_lags(n, step_min)
        active_rolls = self._select_active_rolls(n, step_min)
        self.last_skipped_lags_ = skipped

        feature_names: List[str] = []

        # 1. Лаговые признаки
        for hours in active_lags:
            n_periods = _hours_to_periods(hours, step_min)
            name = f"y_lag_{int(hours)}h"
            data[name] = data["y"].shift(n_periods)
            feature_names.append(name)

        # 2. Скользящие статистики (без утечки: через shift(1))
        shifted = data["y"].shift(1)
        for hours in active_rolls:
            n_periods = _hours_to_periods(hours, step_min)
            mean_name = f"y_mean_{int(hours)}h"
            std_name  = f"y_std_{int(hours)}h"
            data[mean_name] = shifted.rolling(n_periods, min_periods=1).mean()
            data[std_name]  = shifted.rolling(n_periods, min_periods=1).std().fillna(0)
            feature_names += [mean_name, std_name]

            if self.include_min_max:
                min_name = f"y_min_{int(hours)}h"
                max_name = f"y_max_{int(hours)}h"
                data[min_name] = shifted.rolling(n_periods, min_periods=1).min()
                data[max_name] = shifted.rolling(n_periods, min_periods=1).max()
                feature_names += [min_name, max_name]

        # 3. Производные признаки
        data["y_diff"] = data["y"].diff().shift(1).fillna(0)
        feature_names.append("y_diff")

        if self.include_pct_change:
            prev = data["y"].shift(1)
            pct = (data["y"].shift(1) - data["y"].shift(2)) / prev.replace(0, np.nan)
            data["y_pct_change"] = pct.replace([np.inf, -np.inf], 0).fillna(0)
            feature_names.append("y_pct_change")

        # 4. Календарные признаки
        data["hour"]        = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"]  = (data.index.dayofweek >= 5).astype(int)
        feature_names += ["hour", "day_of_week", "is_weekend"]

        if self.include_calendar_cyclic:
            data["hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
            data["hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)
            data["dow_sin"]  = np.sin(2 * np.pi * data.index.dayofweek / 7)
            data["dow_cos"]  = np.cos(2 * np.pi * data.index.dayofweek / 7)
            feature_names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

        # 5. Нормирующие (scale-free) признаки
        has_24h_window = (24 in active_rolls)
        if self.include_zscore and has_24h_window:
            n_24 = _hours_to_periods(24, step_min)
            _mp = min(2, n_24)
            mean_24 = shifted.rolling(n_24, min_periods=_mp).mean()
            std_24  = shifted.rolling(n_24, min_periods=_mp).std()
            data["y_zscore_24h"] = (
                (data["y"].shift(1) - mean_24) / std_24.replace(0, np.nan)
            ).replace([np.inf, -np.inf], 0).fillna(0)

            max_24 = shifted.rolling(n_24, min_periods=1).max()
            data["y_ratio_to_max_24h"] = (
                data["y"].shift(1) / max_24.replace(0, np.nan)
            ).replace([np.inf, -np.inf], 1).fillna(1)

            feature_names += ["y_zscore_24h", "y_ratio_to_max_24h"]

        # 6. Spike-индикаторы.
        # Дают модели сигнал о кластеризации пиков. Без них MSE/MAE-loss
        # всегда тянет прогноз к среднему и пики систематически
        # под-предсказываются (классическая «регрессия к среднему»).
        if self.include_spike_features and has_24h_window:
            n_24 = _hours_to_periods(24, step_min)
            _mp = min(2, n_24)
            mean_24 = shifted.rolling(n_24, min_periods=_mp).mean()
            std_24  = shifted.rolling(n_24, min_periods=_mp).std().fillna(0)

            spike_threshold = mean_24 + self.spike_sigma * std_24
            is_spike_prev = ((shifted > spike_threshold) & (std_24 > 0)).astype(int)
            data["y_is_spike_prev"] = is_spike_prev.fillna(0).astype(int)

            data["y_spike_count_24h"] = (
                is_spike_prev.rolling(n_24, min_periods=1).sum().fillna(0)
            )

            spike_arr = is_spike_prev.values.astype(bool)
            idx = np.arange(len(spike_arr))
            spike_pos = np.where(spike_arr)[0]
            if len(spike_pos):
                last_spike = np.searchsorted(spike_pos, idx, side="right") - 1
                has_prev = last_spike >= 0
                time_since = np.where(has_prev, idx - spike_pos[np.maximum(last_spike, 0)], idx + float(n_24))
            else:
                time_since = idx + float(n_24)
            data["y_periods_since_spike"] = time_since / max(n_24, 1)

            feature_names += [
                "y_is_spike_prev",
                "y_spike_count_24h",
                "y_periods_since_spike",
            ]

        # 7. Праздничные признаки
        # is_holiday=1 в день государственного праздника;
        # is_pre_holiday=1 в день накануне — трафик накануне праздника
        # ведёт себя иначе (вечерний пик смещается/растёт).
        if self.include_holiday_features:
            years = list(set(data.index.year))
            try:
                country_holidays = getattr(_holidays_lib, self.country_code)(years=years)
            except AttributeError:
                country_holidays = {}

            dates = data.index.normalize()
            data["is_holiday"] = dates.map(lambda d: int(d in country_holidays))
            data["is_pre_holiday"] = (
                dates.map(lambda d: int((d + pd.Timedelta(days=1)) in country_holidays))
            )
            feature_names += ["is_holiday", "is_pre_holiday"]

        # 8. Экзогенные метрики (p99_ms, error_rate_pct и любые другие)
        # Для каждой добавляем lag_1h и rolling_mean_1h — без утечки (shift).
        # Для calendar-событий (бинарных флагов) дополнительно добавляем
        # опережающие признаки ahead_1h/3h/6h: модель «видит» предстоящий
        # режим до того, как лаги успели обновиться (фикс step-change).
        for col in self.exog_cols:
            if col not in data.columns:
                if col not in self._warned_exog:
                    logger.warning("exog_col '%s' отсутствует в DataFrame, пропускаем", col)
                    self._warned_exog.add(col)
                continue
            n1 = _hours_to_periods(1, step_min)
            lag_name  = f"{col}_lag_1h"
            roll_name = f"{col}_mean_1h"
            data[lag_name]  = data[col].shift(n1)
            data[roll_name] = data[col].shift(1).rolling(n1, min_periods=1).mean()
            feature_names += [lag_name, roll_name]

            # Опережающие признаки только для бинарных calendar-флагов
            # (значения известны заранее — утечки таргета нет).
            if data[col].dropna().isin([0, 1]).all():
                for ahead_h in [1, 3, 6]:
                    n_ahead = _hours_to_periods(ahead_h, step_min)
                    ahead_name = f"{col}_ahead_{ahead_h}h"
                    # shift(-n) = смотрим вперёд на n периодов
                    data[ahead_name] = data[col].shift(-n_ahead).fillna(0).astype(int)
                    feature_names.append(ahead_name)

        self.last_built_features_ = feature_names
        before = len(data)
        data = data.dropna()
        after = len(data)
        if before > 0:
            logger.debug(
                "FeatureBuilder: %d признаков, %d->%d строк, шаг %.1f мин, "
                "пропущены лаги: %s",
                len(feature_names), before, after, step_min, skipped,
            )
        return data

    @property
    def FEATURE_COLS(self) -> List[str]:
        return list(self.last_built_features_)

    # ------------------------------------------------------------------
    # Адаптация
    # ------------------------------------------------------------------

    def _select_active_lags(
        self, n_rows: int, step_min: float,
    ) -> Tuple[List[float], List[float]]:
        active: List[float] = []
        skipped: List[float] = []
        for hours in self.lag_hours:
            need = _hours_to_periods(hours, step_min)
            if n_rows >= need * self.MIN_HISTORY_MULTIPLIER:
                active.append(hours)
            else:
                skipped.append(hours)
        return active, skipped

    def _select_active_rolls(
        self, n_rows: int, step_min: float,
    ) -> List[float]:
        return [
            h for h in self.roll_hours
            if n_rows >= _hours_to_periods(h, step_min) * self.MIN_HISTORY_MULTIPLIER
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        transformed = self.transform(df)
        cols = [c for c in self.FEATURE_COLS if c in transformed.columns]
        return transformed[cols], transformed["y"]

    def get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(df)
        cols = [c for c in self.FEATURE_COLS if c in transformed.columns]
        return transformed[cols]

    def transform_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ):
        """
        Context-aware feature building: признаки считаются на полном
        объединении (train+val+test), затем срезы возвращаются по позиции.
        """
        full = pd.concat([train, val, test], ignore_index=True).sort_values("ds")
        transformed = self.transform(full)
        cols = [c for c in self.FEATURE_COLS if c in transformed.columns]

        n_test = len(test)
        n_val  = len(val)

        feat_test  = transformed.iloc[-n_test:]
        feat_val   = transformed.iloc[-(n_test + n_val):-n_test]
        feat_train = transformed.iloc[:-(n_test + n_val)]

        def _xy(subset):
            return subset[cols], subset["y"]

        return _xy(feat_train), _xy(feat_val), _xy(feat_test)

    def feature_count(self, df: pd.DataFrame) -> int:
        return len(self.get_X(df.head(min(len(df), 500))).columns)

    def diagnostics(self) -> dict:
        return {
            "n_features": len(self.last_built_features_),
            "feature_names": list(self.last_built_features_),
            "step_minutes": self.last_step_min_,
            "n_rows": self.last_n_rows_,
            "skipped_lags_h": list(self.last_skipped_lags_),
        }


# ---------------------------------------------------------------------------
# Утилиты (обратная совместимость)
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame, label: Optional[str] = None):
    builder = FeatureBuilder()
    if label:
        return builder.get_X_y(df)
    return builder.get_X(df)

def split_train_val_test(
    df: pd.DataFrame,
    test_hours: int = 288,
    val_hours: int = 288,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Хронологическое разбиение без перемешивания.
    Параметр test_hours/val_hours — число ПЕРИОДОВ (не часов!).
    """
    df = df.sort_values("ds").reset_index(drop=True)
    n = len(df)
    split_test = n - test_hours
    split_val  = split_test - val_hours
    train = df.iloc[:split_val].copy()
    val   = df.iloc[split_val:split_test].copy()
    test  = df.iloc[split_test:].copy()
    return train, val, test
