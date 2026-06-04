"""
Детекция концепт-дрейфа на основе ADWIN (Bifet & Gavalda, 2007)
с дополнительным порогом по MAE для медленного дрейфа.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class DriftSignal:
    """Результат проверки дрейфа."""
    triggered: bool
    reason: str
    current_mae: float
    baseline_mae: float
    n_observations: int
    threshold: float


# ---------------------------------------------------------------------------
# ADWIN-based drift detector (Bifet & Gavalda, 2007)
# ---------------------------------------------------------------------------

class ADWINDriftDetector:
    """
    Детектор концепт-дрейфа с двумя независимыми механизмами.

    Механизм 1 — ADWIN (Bifet & Gavalda, 2007):
        Статистический алгоритм с адаптивным окном. Срабатывает при
        статистически значимом изменении среднего ошибки (граница Хёффдинга).
        Хорошо работает при резких сдвигах. При медленном дрейфе даёт
        редкие детекции, поэтому дополняется порогом MAE.
        Ретрейн: confirmation_n детекций ADWIN за последние confirmation_n×10 шагов.

    Механизм 2 — порог MAE (mae_ratio_threshold):
        Ретрейн когда скользящая ошибка превышает baseline_mae в
        mae_ratio_threshold раз. Надёжно ловит медленный дрейф, который
        ADWIN пропускает. 0.0 = отключено.

    Механизм 3 — n_fresh:
        Принудительный ретрейн каждые n_fresh шагов. 0 = отключено.

    Параметры
    ----------
    delta                : уровень доверия ADWIN (0.002 → ≤0.2% ложных срабатываний)
    min_obs              : минимум наблюдений перед первой проверкой
    cooldown_n           : минимум шагов между двумя ретрейнами
    n_fresh              : принудительный ретрейн каждые n_fresh шагов (0 = выкл.)
    confirmation_n       : сколько детекций ADWIN нужно за окно 10×confirmation_n шагов
    mae_ratio_threshold  : ретрейн при current_mae > baseline_mae × ratio (0 = выкл.)
    mae_window           : сколько последних ошибок усреднять для MAE-проверки
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_obs: int = 30,
        cooldown_n: int = 20,
        n_fresh: int = 0,
        confirmation_n: int = 3,
        mae_ratio_threshold: float = 2.0,
        mae_window: int = 20,
    ):
        from river.drift import ADWIN as _ADWIN
        self._adwin = _ADWIN(delta=delta)
        self.delta = delta
        self.min_obs = int(min_obs)
        self.cooldown_n = int(cooldown_n)
        self.n_fresh = int(n_fresh)
        self.confirmation_n = int(confirmation_n)
        self.mae_ratio_threshold = float(mae_ratio_threshold)
        self.mae_window = int(mae_window)

        self._n: int = 0
        self._n_since_retrain: int = 0
        # Окно для подсчёта детекций ADWIN: нужно confirmation_n True за окно
        _window = max(1, confirmation_n) * 10
        self._drift_window: Deque[bool] = deque(maxlen=_window)
        self._baseline_mae: Optional[float] = None
        self._last_errors: list = []

    # ------------------------------------------------------------------

    def set_baseline(self, baseline_errors: np.ndarray) -> None:
        arr = np.abs(np.asarray(baseline_errors, dtype=float))
        self._baseline_mae = float(np.mean(arr)) if arr.size else 0.0
        self._n_since_retrain = 0
        # Засеваем ADWIN обучающими ошибками чтобы он знал «нормальный» уровень
        for e in arr[-100:]:
            self._adwin.update(float(e))

    def reset_after_retrain(self, baseline_errors: np.ndarray) -> None:
        from river.drift import ADWIN as _ADWIN
        self._adwin = _ADWIN(delta=self.delta)
        self._last_errors.clear()
        self._n_since_retrain = 0
        self._drift_window.clear()
        self.set_baseline(baseline_errors)

    def observe(self, y_true: float, y_pred: float) -> None:
        err = abs(float(y_true) - float(y_pred))
        self._adwin.update(err)
        self._last_errors.append(err)
        if len(self._last_errors) > 500:
            self._last_errors.pop(0)
        self._n += 1
        self._n_since_retrain += 1

    def check(self) -> DriftSignal:
        if self._n < self.min_obs:
            return DriftSignal(
                triggered=False, reason="warming_up",
                current_mae=float("nan"),
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n, threshold=float("nan"),
            )

        if self._n_since_retrain < self.cooldown_n:
            return DriftSignal(
                triggered=False, reason="cooldown",
                current_mae=float(np.mean(self._last_errors[-self.mae_window:])) if self._last_errors else float("nan"),
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n, threshold=float("nan"),
            )

        recent = self._last_errors[-self.mae_window:]
        current_mae = float(np.mean(recent)) if recent else float("nan")

        # Механизм 3: принудительный ретрейн каждые n_fresh шагов
        if self.n_fresh > 0 and self._n_since_retrain >= self.n_fresh:
            return DriftSignal(
                triggered=True, reason="n_fresh",
                current_mae=current_mae,
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n, threshold=float(self.n_fresh),
            )

        # Механизм 2: порог MAE — ловит медленный дрейф, который ADWIN пропускает
        if (
            self.mae_ratio_threshold > 0
            and self._baseline_mae
            and self._baseline_mae > 0
            and not np.isnan(current_mae)
            and current_mae >= self._baseline_mae * self.mae_ratio_threshold
        ):
            return DriftSignal(
                triggered=True, reason="mae_ratio",
                current_mae=current_mae,
                baseline_mae=self._baseline_mae,
                n_observations=self._n,
                threshold=self.mae_ratio_threshold,
            )

        # Механизм 1: ADWIN — confirmation_n детекций за последние 10×confirmation_n шагов
        self._drift_window.append(bool(self._adwin.drift_detected))
        if sum(self._drift_window) >= self.confirmation_n:
            self._drift_window.clear()
            return DriftSignal(
                triggered=True, reason="adwin",
                current_mae=current_mae,
                baseline_mae=self._baseline_mae or float("nan"),
                n_observations=self._n, threshold=float(self.confirmation_n),
            )

        return DriftSignal(
            triggered=False, reason="ok",
            current_mae=current_mae,
            baseline_mae=self._baseline_mae or float("nan"),
            n_observations=self._n, threshold=float("nan"),
        )

    @property
    def baseline_mae(self) -> Optional[float]:
        return self._baseline_mae

    @property
    def n_since_retrain(self) -> int:
        return self._n_since_retrain
