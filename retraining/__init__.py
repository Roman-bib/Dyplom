"""
Модуль адаптивного переобучения.

Состав:
  drift_detector  — детекция концепт-дрейфа (ADWIN)
  scheduler       — RetrainScheduler: оркестратор «прогноз → наблюдение
                    → дрейф → переобучение → переключение модели»
"""

from retraining.drift_detector import ADWINDriftDetector, DriftSignal
from retraining.scheduler import RetrainScheduler, RetrainEvent

__all__ = [
    "ADWINDriftDetector",
    "DriftSignal",
    "RetrainScheduler",
    "RetrainEvent",
]
