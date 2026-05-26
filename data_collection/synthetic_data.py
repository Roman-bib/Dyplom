import numpy as np
import pandas as pd
import holidays as holidays_lib
from typing import Optional

_RU_HOLIDAYS = holidays_lib.Russia(years=range(2024, 2027))

def generate_synthetic_traffic(
    days: int = 30,
    freq: str = "1h",
    base_rps: float = 500.0,
    trend_per_day: float = 2.0,
    noise_std: float = 30.0,
    peak_probability: float = 0.02,  # Вероятность именно случайной аномалии
    peak_multiplier: float = 7.5,
    holiday_multiplier: float = 0.8,
    campaign_days_per_month: int = 2,
    campaign_multiplier: float = 2.5,
    missing_rate: float = 0.03,      # ТЗ: доля пропусков (NaN)
    duplicate_rate: float = 0.01,    # ТЗ: доля дубликатов строк
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)

    # Расчет периодов
    offset = pd.tseries.frequencies.to_offset(freq)
    periods = int(pd.Timedelta(f"{days}D") / offset)
    timestamps = pd.date_range(start="2025-01-01", periods=periods, freq=freq)

    t = np.arange(periods)
    hours = timestamps.hour.values
    weekday = timestamps.dayofweek.values 
    dates = timestamps.normalize() # Для быстрой векторизации настроек дня

    # --- 1. Тренд ---
    hours_per_day = int(pd.Timedelta("1D") / offset)
    trend = trend_per_day * (t / hours_per_day)

    # --- 2. Суточная сезонность (Исправлено: cos вместо sin для точечных пиков) ---
    daily = (
        0.6 * np.cos(2 * np.pi * (hours - 9) / 24)
        + 0.4 * np.cos(2 * np.pi * (hours - 19) / 24)
    )
    daily_amplitude = base_rps * 0.45
    daily_component = daily_amplitude * daily

    # --- 3. Недельная сезонность ---
    weekend_mask = (weekday >= 5).astype(float)
    weekly_component = -base_rps * 0.3 * weekend_mask

    # --- 4. Шум ---
    noise = rng.normal(0, noise_std, size=periods)

    # --- 5. Периодические пики (Предсказуемые всплески) ---
    peak_hour1, peak_hour2 = 10, 19
    peak_width = 2
    hour_float = timestamps.hour + timestamps.minute / 60.0
    is_weekday = (weekday < 5).astype(float)

    periodic_peaks = (
        base_rps * (peak_multiplier - 1) * is_weekday * (
            np.exp(-0.5 * ((hour_float - peak_hour1) / (peak_width / 2)) ** 2) * 0.7
            + np.exp(-0.5 * ((hour_float - peak_hour2) / (peak_width / 2)) ** 2)
        )
        + base_rps * (peak_multiplier - 1) * 0.5
        * (weekday == 4).astype(float)
        * np.exp(-0.5 * ((hour_float - 18) / 1.0) ** 2)
    )

    # --- 6. Праздники ---
    holiday_mask = np.array([d in _RU_HOLIDAYS for d in dates.date])
    holiday_component = base_rps * holiday_multiplier * holiday_mask

    # --- 7. Маркетинговые акции (Векторизовано и ускорено) ---
    campaign_component = np.zeros(periods)
    is_campaign = np.zeros(periods, dtype=int)
    
    unique_months = pd.Index(timestamps).to_period("M").unique()
    
    for month in unique_months:
        month_mask = (timestamps.to_period("M") == month)
        month_dates = dates[month_mask].unique()
        workdays = month_dates[month_dates.dayofweek < 5]
        
        if len(workdays) >= campaign_days_per_month:
            chosen_days = rng.choice(workdays, size=campaign_days_per_month, replace=False)
            # Быстрая маска через isin без циклов по таймстемпам
            campaign_day_mask = dates.isin(chosen_days)
            
            is_campaign[campaign_day_mask] = 1
            campaign_component[campaign_day_mask] = (
                (base_rps + trend[campaign_day_mask]) * (campaign_multiplier - 1)
            )

    # --- 8. Аномалии (ТЗ: Добавлены как спады, так и выбросы) ---
    anomaly_roll = rng.random(size=periods)
    anomaly_values = np.zeros(periods)
    
    # 1.5% — резкие пики (выбросы), 0.5% — просадки трафика (аварии)
    spike_mask = anomaly_roll < (peak_probability * 0.75)
    drop_mask = (anomaly_roll >= (peak_probability * 0.75)) & (anomaly_roll < peak_probability)
    
    anomaly_values[spike_mask] = (base_rps + trend[spike_mask]) * rng.uniform(2.0, 4.0, size=spike_mask.sum())
    anomaly_values[drop_mask] = -(base_rps + trend[drop_mask]) * rng.uniform(0.5, 0.9, size=drop_mask.sum())

    # --- Сборка целевой метрики RPS ---
    rps = (
        base_rps + trend + daily_component + weekly_component 
        + noise + periodic_peaks + holiday_component + campaign_component + anomaly_values
    )
    rps = np.clip(rps, 0, None)

    # --- 9. Контекстные метрики: p99 латентность и ошибки (Исправлен лимит) ---
    capacity_rps = base_rps * peak_multiplier * 0.7
    utilization = rps / capacity_rps
    
    # Подняли лимит до 0.99, чтобы p99_ms мог долетать до верхних лимитов при перегрузках
    utilization_clipped = np.clip(utilization, 0, 0.99) 
    p99_ms = 50.0 / (1.0 - utilization_clipped) + rng.normal(0, 3, size=periods)
    p99_ms = np.clip(p99_ms, 30, 8000)

    error_rate_pct = np.where(
        utilization > 0.90,
        np.clip((utilization - 0.90) * 150, 0, 100),
        0.0,
    ) + np.clip(rng.normal(0, 0.05, size=periods), 0, None)
    error_rate_pct = np.clip(error_rate_pct, 0, 100)

    # Формируем базовый датасет
    df = pd.DataFrame({
        "ds": timestamps,
        "y": rps,
        "p99_ms": p99_ms,
        "error_rate_pct": error_rate_pct,
        "is_campaign": is_campaign,
    })

    # --- 10. ТЗ: Внедрение пропусков (NaN) ---
    if missing_rate > 0:
        for col in ["y", "p99_ms", "error_rate_pct"]:
            mask = rng.random(size=len(df)) < missing_rate
            df.loc[mask, col] = np.nan

    # --- 11. ТЗ: Внедрение дубликатов ---
    if duplicate_rate > 0:
        num_dups = int(len(df) * duplicate_rate)
        if num_dups > 0:
            dup_indices = rng.choice(df.index, size=num_dups, replace=True)
            df_dups = df.loc[dup_indices].copy()
            # Слегка смешиваем, чтобы дубликаты шли друг за другом или хаотично
            df = pd.concat([df, df_dups], ignore_index=True).sort_values("ds").reset_index(drop=True)

    return df