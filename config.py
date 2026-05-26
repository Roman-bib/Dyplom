# =============================================================================
# Конфигурация системы прогнозирования пиковых нагрузок (ВКР)
# =============================================================================

# Параметры данных
DEFAULT_STEP = "5min"               # шаг дискретизации: 1 точка каждые 5 минут

# --- Обучение ---
MODEL_SAVE_DIR = "./saved_models"

# Горизонты прогнозирования (в числе периодов при DEFAULT_STEP="5min"):
#   15 мин = 3 периода, 30 мин = 6, 60 мин = 12
FORECAST_HORIZONS_PERIODS = [3, 6, 12]

# Разбиение данных (в периодах, при DEFAULT_STEP="5min"):
#   24ч * 12 периодов/ч = 288 периодов
SPLIT_TEST_PERIODS = 288           # 24 часа тестовых данных
SPLIT_VAL_PERIODS = 288            # 24 часа валидационных данных

# Для обратной совместимости (используются в main.py как hours, но фактически — периоды)
SPLIT_TEST_HOURS = SPLIT_TEST_PERIODS
SPLIT_VAL_HOURS = SPLIT_VAL_PERIODS

# XGBoost гиперпараметры
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_EARLY_STOPPING = 20

# --- Детекция пиков ---
PEAK_PERCENTILE = 95.0              # квантиль скользящего окна
PEAK_WINDOW_HOURS = 24              # окно (периодов) для расчёта перцентиля
PEAK_RECOMPUTE_EVERY = 12           # пересчёт порога каждые N шагов

# Пороги уровней алертов (доля от threshold)
ALERT_WARNING_RATIO = 0.70          # ≥70% от порога → warning
ALERT_CRITICAL_RATIO = 0.85         # ≥85% от порога → critical

# --- Масштабирование ---
# При 5-минутной детализации и типичном трафике grafana-deploy:
TARGET_LOAD_PER_REPLICA = 10.0      # RPS на реплику (бэкенд маленький, 10 RPS на инстанс)
MIN_REPLICAS = 1
MAX_REPLICAS = 10
SCALE_DOWN_BACKOFF = 300            # задержка уменьшения реплик (сек) = 5 минут

# --- ADWIN детектор концепт-дрейфа (глава 2.7.2 ВКР) ---
# delta=0.002 → P(ложное срабатывание) ≤ 0.2% (стандарт Bifet & Gavalda, 2007)
ADWIN_DELTA = 0.002
ADWIN_MIN_OBS = 30                  # минимум наблюдений перед первой проверкой
ADWIN_COOLDOWN_N = 20               # минимум шагов между retrain-событиями
ADWIN_CONFIRMATION_N = 10           # шагов подряд с дрейфом до ретрейна (отсекает пики)
# N_fresh: принудительный retrain каждые N шагов (блок-схема Рисунок 2.7).
# 0 = отключить. При DEFAULT_STEP=5min: 1440 шагов ≈ 5 суток.
ADWIN_N_FRESH = 288

# --- Порог переобучения (глава 2.6.4 ВКР) ---
MAPE_RETRAIN_THRESHOLD = 20.0       # при MAPE > 20% инициировать переобучение

# --- Праздники и события ---
PROPHET_USE_HOLIDAYS = True         # передавать российские праздники в Prophet
PROPHET_COUNTRY_CODE = "RU"        # ISO-код страны для add_country_holidays

# --- Экзогенные метрики (дополнительные признаки для модели) ---
# Колонки, которые FeatureBuilder добавит как lag_1h + mean_1h признаки.
# При загрузке из Prometheus — заполняются отдельными запросами;
# при синтетике — генерируются автоматически в synthetic_data.py.
EXOG_COLS = ["is_campaign", "is_promo"]

# --- Детектор аномальных пиков (Isolation Forest на остатке r_t) ---
IF_CONTAMINATION = 0.05             # ожидаемая доля аномалий в обучающей выборке
IF_SAFETY_FACTOR = 1.2              # коэффициент запаса реплик при аномальном пике

# --- Синтетические данные (для демо без Prometheus) ---
SYNTHETIC_DAYS = 30
SYNTHETIC_FREQ = "5min"             # шаг синтетики = шагу реального сбора
SYNTHETIC_BASE_RPS = 5.0            # RPS типичен для небольшого тестового стека
SYNTHETIC_NOISE_STD = 1.0
SYNTHETIC_PEAK_PROB = 0.015
SYNTHETIC_PEAK_MULTIPLIER = 4.0
