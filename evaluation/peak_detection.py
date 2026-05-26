"""
Детекция пиковых нагрузок (ВКР).

Метод порога: adaptive_percentile — 95-й перцентиль скользящего окна 24 точки,
пересчёт каждые recompute_every шагов в detect_series().
ResidualAnomalyDetector — изолирующий лес на остатке r_t для классификации
природы пика (сезонный vs аномальный).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import joblib

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

Severity = type("Severity", (), {})  # строки: "ok" | "info" | "warning" | "critical" | "exceeded"

_SEVERITY_RANK = {
    "ok": 0, "info": 1, "warning": 2, "critical": 3, "exceeded": 4,
}


@dataclass
class PeakEvent:
    timestamp: pd.Timestamp
    current_rps: float
    predicted_rps: float
    threshold: float
    severity: str
    recommended_replicas: int
    novelty: bool = False

    def __str__(self):
        icon = {
            "ok": "[OK]", "info": "[i]",
            "warning": "[!]", "critical": "[!!]", "exceeded": "[X]",
        }[self.severity]
        novelty_tag = " [NOVELTY]" if self.novelty else ""
        return (
            f"{icon} [{self.severity.upper()}]{novelty_tag} {self.timestamp} | "
            f"RPS текущий={self.current_rps:.0f}, прогноз={self.predicted_rps:.0f}, "
            f"порог={self.threshold:.0f} | реплик={self.recommended_replicas}"
        )

    @property
    def is_peak(self) -> bool:
        return self.severity in ("warning", "critical", "exceeded")


# ---------------------------------------------------------------------------
# Детектор
# ---------------------------------------------------------------------------

class PeakDetector:
    """
    Определяет, является ли прогнозируемое значение RPS пиковым,
    и рекомендует количество реплик.

    Метод порога: adaptive_percentile.
    Порог = percentile-й перцентиль последних window точек истории.
    В detect_series() пересчитывается каждые recompute_every шагов.

    Parameters
    ----------
    percentile             : квантиль (0..100), по умолчанию 95
    window                 : размер скользящего окна в периодах (по умолчанию 24)
    target_rps_per_replica : нагрузка RPS на одну реплику
    min_replicas           : минимальное число реплик
    max_replicas           : максимальное число реплик
    warning_ratio          : доля от порога для уровня warning (0.70)
    critical_ratio         : доля от порога для уровня critical (0.85)
    spike_growth           : рост predicted/current для уровня info (1.2)
    """

    def __init__(
        self,
        percentile: float = 95.0,
        window: int = 24,
        target_rps_per_replica: float = 10.0,
        min_replicas: int = 1,
        max_replicas: int = 10,
        warning_ratio: float = 0.70,
        critical_ratio: float = 0.85,
        spike_growth: float = 1.2,
    ):
        self.percentile = percentile
        self.window = window
        self.target_rps = float(target_rps_per_replica)
        self.min_replicas = int(min_replicas)
        self.max_replicas = int(max_replicas)
        self.warning_ratio = float(warning_ratio)
        self.critical_ratio = float(critical_ratio)
        self.spike_growth = float(spike_growth)

        self._threshold: Optional[float] = None
        self._fit_history: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, historical_rps: pd.Series) -> "PeakDetector":
        """Вычисляет начальный порог по историческим данным. Вызывать перед detect()."""
        s = pd.Series(historical_rps).dropna().astype(float)
        if s.empty:
            raise ValueError("historical_rps пустой — невозможно посчитать порог")

        self._fit_history = s
        tail = s.tail(self.window)
        self._threshold = float(np.percentile(tail.values, self.percentile))
        return self

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Вызовите .fit() перед обращением к threshold")
        return self._threshold

    # ------------------------------------------------------------------
    # Single-point detection
    # ------------------------------------------------------------------

    def detect(
        self,
        predicted_rps: float,
        current_rps: float = 0.0,
        timestamp: Optional[pd.Timestamp] = None,
        threshold_override: Optional[float] = None,
        novelty: bool = False,
        novelty_safety_factor: float = 1.0,
    ) -> PeakEvent:
        """
        Классифицирует прогнозируемую нагрузку.

        Уровни (монотонные):
          exceeded — predicted ≥ threshold
          critical — predicted ≥ threshold * critical_ratio
          warning  — predicted ≥ threshold * warning_ratio
          info     — predicted ≥ current * spike_growth
          ok       — иначе
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        threshold = threshold_override if threshold_override is not None else self.threshold

        if threshold <= 0:
            severity = "ok"
        else:
            ratio = predicted_rps / threshold
            if ratio >= 1.0:
                severity = "exceeded"
            elif ratio >= self.critical_ratio:
                severity = "critical"
            elif ratio >= self.warning_ratio:
                severity = "warning"
            elif current_rps > 0 and predicted_rps > current_rps * self.spike_growth:
                severity = "info"
            else:
                severity = "ok"

        replicas = self._calculate_replicas(
            predicted_rps * novelty_safety_factor if novelty else predicted_rps
        )

        return PeakEvent(
            timestamp=timestamp,
            current_rps=float(current_rps),
            predicted_rps=float(predicted_rps),
            threshold=float(threshold),
            severity=severity,
            recommended_replicas=replicas,
            novelty=novelty,
        )

    # ------------------------------------------------------------------
    # Series detection
    # ------------------------------------------------------------------

    def detect_series(
        self,
        rps_series: pd.Series,
        predicted_series: pd.Series,
        recompute_every: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Применяет детекцию ко всему ряду.

        Parameters
        ----------
        recompute_every : пересчитывать порог каждые N точек по последнему
                          окну self.window (адаптивный режим).

        Returns
        -------
        DataFrame: timestamp, rps, predicted, threshold,
                   severity, recommended_replicas, is_peak
        """
        records = []
        rolling_threshold = self.threshold
        history_buffer: list = list(self._fit_history.values) \
            if self._fit_history is not None else []

        for i, (ts, cur, pred) in enumerate(zip(
            rps_series.index, rps_series.values, predicted_series.values,
        )):
            if (
                recompute_every
                and i > 0
                and i % recompute_every == 0
                and len(history_buffer) >= self.window
            ):
                tail = history_buffer[-self.window:]
                rolling_threshold = float(np.percentile(tail, self.percentile))

            event = self.detect(
                predicted_rps=float(pred),
                current_rps=float(cur),
                timestamp=pd.Timestamp(ts),
                threshold_override=rolling_threshold,
            )
            records.append({
                "timestamp": event.timestamp,
                "rps": event.current_rps,
                "predicted": event.predicted_rps,
                "threshold": event.threshold,
                "severity": event.severity,
                "recommended_replicas": event.recommended_replicas,
                "is_peak": event.is_peak,
            })

            history_buffer.append(float(cur))

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Replicas
    # ------------------------------------------------------------------

    def _calculate_replicas(self, predicted_rps: float) -> int:
        """Минимальное число реплик, удовлетворяющее load: ceil(load/target)."""
        if self.target_rps <= 0:
            return self.min_replicas
        desired = int(np.ceil(max(0.0, predicted_rps) / self.target_rps))
        return int(np.clip(desired, self.min_replicas, self.max_replicas))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, events_df: pd.DataFrame) -> dict:
        """Сводная статистика по серии детекций."""
        total = len(events_df)
        peaks = int(events_df["is_peak"].sum()) if total else 0

        sev_counts_raw = (
            events_df["severity"].value_counts().to_dict() if total else {}
        )
        sev_counts = {
            sev: int(sev_counts_raw.get(sev, 0))
            for sev in ("ok", "info", "warning", "critical", "exceeded")
        }

        return {
            "total_points": int(total),
            "peaks_detected": peaks,
            "peak_ratio_pct": round(peaks / total * 100, 2) if total else 0.0,
            "threshold": round(self.threshold, 2) if self._threshold else None,
            "severity_counts": sev_counts,
        }


class ResidualAnomalyDetector:
    """
    Детектор аномальных пиков на основе изолирующего леса (Isolation Forest).

    Обучается на остатке r_t = y_t - T_t - S_t обучающей выборки,
    полученном из TimeSeriesCleaner.transform(). Классифицирует пик
    как аномальный, если его остаток изолируется за малое число разбиений.

    Parameters
    ----------
    contamination : ожидаемая доля аномалий (0..0.5)
    n_estimators  : число деревьев изоляции
    """

    _FILENAME = "if_anomaly.pkl"

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None

    def fit(self, residuals: np.ndarray) -> "ResidualAnomalyDetector":
        """Обучает IF на остатках обучающей выборки."""
        from sklearn.ensemble import IsolationForest
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(np.asarray(residuals).reshape(-1, 1))
        return self

    def predict(self, residuals: np.ndarray) -> np.ndarray:
        """Возвращает булев массив: True = аномальный пик."""
        if self._model is None:
            return np.zeros(len(residuals), dtype=bool)
        return self._model.predict(
            np.asarray(residuals).reshape(-1, 1)
        ) == -1

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self._model, os.path.join(directory, self._FILENAME))

    @classmethod
    def load(cls, directory: str) -> "ResidualAnomalyDetector":
        obj = cls()
        obj._model = joblib.load(os.path.join(directory, cls._FILENAME))
        return obj


def plot_peaks(
    events_df: pd.DataFrame,
    summary: dict,
    save_path: str,
    max_replicas: int = 10,
) -> str:
    """
    График детекции пиков: нагрузка + прогноз + порог (верхний subplot)
    и рекомендованное число реплик (нижний subplot).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    timestamps = pd.to_datetime(events_df["timestamp"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(timestamps, events_df["rps"], color="steelblue",
            linewidth=1.2, label="Фактическая нагрузка", zorder=2)
    ax.plot(timestamps, events_df["predicted"], color="orange",
            linewidth=1.2, linestyle="--", label="Прогноз", zorder=2)

    threshold = summary.get("threshold", 0)
    if threshold:
        ax.axhline(threshold, color="crimson", linestyle=":",
                   linewidth=1.0, label=f"Порог ({threshold:.0f} RPS)", zorder=1)

    warn_mask = events_df["severity"] == "warning"
    crit_mask = events_df["severity"] == "critical"
    if warn_mask.any():
        ax.scatter(timestamps[warn_mask], events_df["rps"][warn_mask],
                   color="gold", s=40, zorder=5, label="Пик: warning")
    if crit_mask.any():
        ax.scatter(timestamps[crit_mask], events_df["rps"][crit_mask],
                   color="crimson", s=60, zorder=5, marker="^", label="Пик: critical")

    ax.set_ylabel("Нагрузка (RPS)")
    ax.set_title("Детекция пиков нагрузки на тестовой выборке")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.step(timestamps, events_df["recommended_replicas"],
             color="steelblue", linewidth=1.2, where="post")
    ax2.fill_between(timestamps, events_df["recommended_replicas"],
                     step="post", alpha=0.2, color="steelblue")
    ax2.set_ylabel("Реплики")
    ax2.set_xlabel("Время")
    ax2.set_ylim(0, max_replicas + 1)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
