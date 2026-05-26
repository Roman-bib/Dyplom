"""
Менеджер реактивного/проактивного масштабирования для веб-сервиса.

Поведение:
  scale UP   — мгновенно (нет ожидания, ошибка под нагрузкой дороже).
  scale DOWN — отложено на scale_down_backoff секунд (защита от
               частых колебаний при шумных прогнозах).

После аудита (см. AUDIT_REPORT.md, шаг 9):
  • Заменена формула расчёта реплик: было `int(load/target) + 1`,
    что давало 11 реплик при ровно `target*10` нагрузке (off-by-one).
    Стало math.ceil(load / target) — математически корректно.
  • print заменён на logging — можно отдельно настраивать уровень
    и направление логов в production.
  • Добавлен __del__ / cancel(), чтобы Timer не висел после удаления
    объекта (потенциальная утечка потоков).
"""

from __future__ import annotations

import logging
import math
from threading import Timer
from typing import Optional

from config import (
    TARGET_LOAD_PER_REPLICA,
    MIN_REPLICAS,
    MAX_REPLICAS,
    SCALE_DOWN_BACKOFF,
)


logger = logging.getLogger(__name__)


class MomentumScaler:
    def __init__(
        self,
        target_load_per_replica: float = TARGET_LOAD_PER_REPLICA,
        min_replicas: int = MIN_REPLICAS,
        max_replicas: int = MAX_REPLICAS,
        scale_down_backoff: float = SCALE_DOWN_BACKOFF,
    ):
        if target_load_per_replica <= 0:
            raise ValueError("target_load_per_replica must be > 0")
        if min_replicas < 1:
            raise ValueError("min_replicas must be ≥ 1")
        if max_replicas < min_replicas:
            raise ValueError("max_replicas must be ≥ min_replicas")

        self.target_load = float(target_load_per_replica)
        self.min = int(min_replicas)
        self.max = int(max_replicas)
        self.scale_down_backoff = float(scale_down_backoff)
        self.current_replicas = self.min
        self.scale_timer: Optional[Timer] = None

    # ------------------------------------------------------------------

    def calculate_desired_replicas(self, predicted_load: float) -> int:
        """
        Минимальное число реплик, удовлетворяющее load:
            desired = ceil(load / target)
        затем clip в [min, max].
        """
        if predicted_load <= 0:
            return self.min
        desired = math.ceil(predicted_load / self.target_load)
        return max(self.min, min(self.max, desired))

    def scale(self, predicted_load: float) -> int:
        """
        Принимает решение о масштабировании.
        Возвращает целевое число реплик (для тестов и логов).
        """
        desired = self.calculate_desired_replicas(predicted_load)

        if desired > self.current_replicas:
            self._cancel_pending()
            old = self.current_replicas
            self.current_replicas = desired
            logger.info(
                "Scale UP: %d -> %d replicas (load=%.1f)",
                old, desired, predicted_load,
            )
        elif desired < self.current_replicas:
            self._cancel_pending()
            self.scale_timer = Timer(
                self.scale_down_backoff,
                self._scale_down,
                args=[desired],
            )
            self.scale_timer.daemon = True   # не блокирует выход программы
            self.scale_timer.start()
            logger.info(
                "Scale DOWN scheduled in %.0fs: %d -> %d replicas",
                self.scale_down_backoff, self.current_replicas, desired,
            )
        else:
            logger.debug(
                "Replicas optimal: %d (load=%.1f)",
                self.current_replicas, predicted_load,
            )

        return desired

    # ------------------------------------------------------------------

    def _scale_down(self, replicas: int) -> None:
        old = self.current_replicas
        self.current_replicas = replicas
        logger.info("Scale DOWN executed: %d -> %d replicas", old, replicas)

    def _cancel_pending(self) -> None:
        if self.scale_timer is not None:
            self.scale_timer.cancel()
            self.scale_timer = None

    def close(self) -> None:
        """Освобождает ресурсы. Вызывать перед удалением объекта."""
        self._cancel_pending()

    def __del__(self):                              # safety net
        try:
            self._cancel_pending()
        except Exception:
            pass


class HybridScaler(MomentumScaler):
    """
    Гибридная схема: proactive + reactive HPA-fallback.

    Принимает два сигнала:
      • proactive_load — прогнозируемая нагрузка от ML-модели;
      • observed_load  — фактическая текущая нагрузка.

    Решение принимается по большему из двух предложенных значений
    реплик: ML-прогноз + страхующий реактивный механизм по текущей
    загрузке. Так система устойчива к ошибкам прогноза: если ML
    «промахнулся», реактивная составляющая всё равно увеличит
    реплики при превышении target_load.

    Это аналог архитектуры AWS Predictive Scaling
    (predictive_estimate + target_tracking) и стандарт production-
    систем проактивного автомасштабирования.
    """

    def scale_hybrid(
        self, proactive_load: float, observed_load: float,
    ) -> tuple:
        """
        Возвращает (n_desired, n_proactive, n_reactive).

        n_desired = max(n_proactive, n_reactive) — берём наибольшее,
        чтобы реактивный путь страховал.
        """
        n_proactive = self.calculate_desired_replicas(proactive_load)
        n_reactive  = self.calculate_desired_replicas(observed_load)
        n_desired   = max(n_proactive, n_reactive)
        # Используем общую логику scale() для применения и backoff
        if n_desired > self.current_replicas:
            self._cancel_pending()
            self.current_replicas = n_desired
            logger.info(
                "Hybrid scale UP -> %d (proactive=%d, reactive=%d, "
                "load_pred=%.1f, load_obs=%.1f)",
                n_desired, n_proactive, n_reactive,
                proactive_load, observed_load,
            )
        elif n_desired < self.current_replicas:
            self._cancel_pending()
            self.scale_timer = Timer(
                self.scale_down_backoff,
                self._scale_down,
                args=[n_desired],
            )
            self.scale_timer.daemon = True
            self.scale_timer.start()
            logger.info(
                "Hybrid scale DOWN scheduled in %.0fs: %d -> %d",
                self.scale_down_backoff, self.current_replicas, n_desired,
            )
        return n_desired, n_proactive, n_reactive
