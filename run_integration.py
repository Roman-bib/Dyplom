"""
Сквозной интеграционный скрипт для работы с реальным Prometheus.
Стек: github.com/artemonsh/grafana-deploy

Порядок работы:
  1. Запустить стек:       docker-compose up -d  (в папке grafana-deploy)
  2. Запустить нагрузку:   locust -f simulation/locustfile.py --host=http://localhost:8080
                           (дождаться минимум 1 часа трафика, лучше — нескольких дней)
  3. Обучить модель:       python run_integration.py train
  4. Прогноз:              python run_integration.py forecast
  5. Непрерывный мониторинг: python run_integration.py monitor

Флаги:
  --days N       глубина истории (по умолчанию из config.py)
  --step STR     шаг данных: "1min", "5min", "1h" и т.д.
  --no-neural-prophet  пропустить NeuralProphet (ускоряет обучение)
  --save-plots   сохранить графики в saved_models/
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

import config
from data_collection.prometheus_connector import (
    check_connection,
    fetch_rps,
    fetch_latency_p99,
)
from preprocessing.feature_engineering import FeatureBuilder, split_train_val_test
from models.xgboost_model import (
    train_xgboost, predict_xgboost,
    get_confidence_interval, feature_importance,
)
from models.model_comparison import ModelComparison
from evaluation.metrics import evaluate, plot_forecast, plot_all_forecasts
from evaluation.peak_detection import PeakDetector
from scaling.scaler import MomentumScaler


# ---------------------------------------------------------------------------
# Проверка готовности стека
# ---------------------------------------------------------------------------

def assert_prometheus_ready(url: str = config.PROMETHEUS_URL) -> None:
    """Завершает скрипт если Prometheus недоступен."""
    print(f"Проверка Prometheus ({url})...", end=" ", flush=True)
    if check_connection(url):
        print("OK")
    else:
        print("НЕДОСТУПЕН")
        print(
            "\nУбедитесь что стек запущен:\n"
            "  cd grafana-deploy && docker-compose up -d\n"
            "Prometheus должен быть доступен на http://localhost:9090"
        )
        sys.exit(1)


def check_data_volume(df: pd.DataFrame, min_points: int = 100) -> None:
    """Предупреждает если данных мало для обучения."""
    n = len(df)
    if n < min_points:
        print(
            f"\n[!] ВНИМАНИЕ: получено только {n} точек (минимум рекомендуется {min_points}).\n"
            "  Запустите Locust и дайте ему поработать хотя бы 1 час.\n"
            "  Команда: locust -f simulation/locustfile.py --host=http://localhost:8080\n"
        )
    else:
        print(f"Данных достаточно: {n} точек")


# ---------------------------------------------------------------------------
# Режим: обучение
# ---------------------------------------------------------------------------

def mode_train(args) -> None:
    assert_prometheus_ready()

    print(f"\nЗагрузка данных из Prometheus (последние {args.days} дней, шаг {args.step})...")
    df = fetch_rps(
        days_ago=args.days,
        step=args.step,
        prometheus_url=config.PROMETHEUS_URL,
        token=config.PROMETHEUS_ACCESS_TOKEN,
    )
    print(f"RPS: min={df['y'].min():.3f}  max={df['y'].max():.3f}  mean={df['y'].mean():.3f}")
    check_data_volume(df, min_points=200)

    train, val, test = split_train_val_test(
        df,
        test_hours=config.SPLIT_TEST_PERIODS,
        val_hours=config.SPLIT_VAL_PERIODS,
    )
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}\n")

    if len(train) < 50:
        print("Недостаточно данных для обучения. Нужно больше трафика.")
        sys.exit(1)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # --- Сравнение моделей ---
    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(train, val, test, include_prophet=not args.no_neural_prophet)
    comparator.save_best()

    # --- XGBoost: доверительный интервал ---
    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    xgb_path = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    if not os.path.exists(xgb_path):
        print("XGBoost не найден — обучаем отдельно...")
        train_xgboost(X_train, y_train, X_val, y_val, save_path=xgb_path)

    model = joblib.load(xgb_path)
    preds = predict_xgboost(model, X_test)

    print("\nОбучение квантильных моделей...")
    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test,
        save_dir=config.MODEL_SAVE_DIR,
    )

    # --- Важность признаков ---
    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nВажность признаков (XGBoost gain):")
    print(imp_df.to_string(index=False))

    # --- График ---
    plot_kwargs = {}
    if args.save_plots:
        plot_kwargs["save_path"] = os.path.join(
            config.MODEL_SAVE_DIR, "forecast_prometheus.png"
        )
    plot_forecast(
        train, val, test, preds, "XGBoost (реальные данные)",
        lower=lower, upper=upper, **plot_kwargs
    )

    # --- Детекция пиков ---
    _run_peak_detection(train["y"], y_test, preds)

    print(f"\nМодели сохранены в {config.MODEL_SAVE_DIR}/")


# ---------------------------------------------------------------------------
# Режим: прогноз (разовый)
# ---------------------------------------------------------------------------

def mode_forecast(args) -> None:
    assert_prometheus_ready()

    xgb_path = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    if not os.path.exists(xgb_path):
        print("Модель не найдена. Сначала запустите: python run_integration.py train")
        sys.exit(1)

    model = joblib.load(xgb_path)
    builder = FeatureBuilder()

    print(f"Загрузка свежих данных (последние 2 дня, шаг {args.step})...")
    df = fetch_rps(
        days_ago=2,
        step=args.step,
        prometheus_url=config.PROMETHEUS_URL,
    )

    if len(df) < 30:
        print("Недостаточно данных для прогноза.")
        sys.exit(1)

    X = builder.get_X(df)
    if len(X) == 0:
        print("Не удалось построить признаки (нужно больше данных).")
        sys.exit(1)

    # Прогноз на последнюю строку (следующий период)
    last_row = X.iloc[[-1]]
    predicted_rps = float(predict_xgboost(model, last_row)[0])
    current_rps = float(df["y"].iloc[-1])

    print(f"\nТекущий RPS:    {current_rps:.3f}")
    print(f"Прогноз RPS:    {predicted_rps:.3f}")

    # Детекция пика
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=config.TARGET_LOAD_PER_REPLICA,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
        warning_ratio=config.ALERT_WARNING_RATIO,
        critical_ratio=config.ALERT_CRITICAL_RATIO,
    )
    detector.fit(df["y"])
    event = detector.detect(
        predicted_rps=predicted_rps,
        current_rps=current_rps,
        timestamp=df["ds"].iloc[-1],
    )
    print(f"\n{event}")

    # Рекомендация масштабирования
    scaler = MomentumScaler()
    scaler.scale(predicted_rps)


# ---------------------------------------------------------------------------
# Режим: непрерывный мониторинг
# ---------------------------------------------------------------------------

def mode_monitor(args) -> None:
    assert_prometheus_ready()

    xgb_path = os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl")
    if not os.path.exists(xgb_path):
        print("Модель не найдена. Запустите: python run_integration.py train")
        sys.exit(1)

    model = joblib.load(xgb_path)
    builder = FeatureBuilder()
    scaler = MomentumScaler()
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=config.TARGET_LOAD_PER_REPLICA,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
        warning_ratio=config.ALERT_WARNING_RATIO,
        critical_ratio=config.ALERT_CRITICAL_RATIO,
    )

    interval = args.interval
    iteration = 0
    print(f"Мониторинг запущен (интервал={interval}с). Ctrl+C для остановки.\n")

    while True:
        try:
            # Загружаем данные за последние 48ч
            df = fetch_rps(
                days_ago=2,
                step=args.step,
                prometheus_url=config.PROMETHEUS_URL,
            )

            if len(df) < 30:
                print(f"[{iteration}] Недостаточно данных, жду...")
                time.sleep(interval)
                continue

            detector.fit(df["y"])
            X = builder.get_X(df)

            if len(X) == 0:
                time.sleep(interval)
                continue

            predicted_rps = float(predict_xgboost(model, X.iloc[[-1]])[0])
            current_rps   = float(df["y"].iloc[-1])

            event = detector.detect(
                predicted_rps=predicted_rps,
                current_rps=current_rps,
                timestamp=pd.Timestamp.now(),
            )
            print(f"[iter={iteration:4d}] {event}")
            scaler.scale(predicted_rps)

            # Периодически проверяем качество модели
            if iteration > 0 and iteration % 10 == 0:
                _check_mape(model, builder, df)

            iteration += 1
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nМониторинг остановлен.")
            break
        except Exception as exc:
            print(f"Ошибка итерации {iteration}: {exc}")
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _run_peak_detection(train_y: pd.Series, y_test: pd.Series, preds: np.ndarray) -> None:
    detector = PeakDetector(
        method=config.PEAK_METHOD,
        k=config.PEAK_K,
        target_rps_per_replica=config.TARGET_LOAD_PER_REPLICA,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
    )
    detector.fit(train_y)

    predicted_series = pd.Series(preds, index=y_test.index)
    events_df = detector.detect_series(y_test, predicted_series)
    summary = detector.summary(events_df)

    print(f"\n--- Детекция пиков ---")
    print(f"  Метод: {config.PEAK_METHOD}, порог={summary['threshold']:.3f} RPS")
    print(f"  Пиков обнаружено: {summary['peaks_detected']} / {summary['total_points']} "
          f"({summary['peak_ratio_pct']}%)")
    print(f"  Уровни severity:  {summary['severity_counts']}")

    if events_df["is_peak"].any():
        peaks_df = events_df[events_df["is_peak"]]
        print(f"\n  Первые 5 пиковых моментов:")
        for _, row in peaks_df.head(5).iterrows():
            print(f"    {row['timestamp']}  "
                  f"RPS={row['rps']:.3f}  прогноз={row['predicted']:.3f}  "
                  f"[{row['severity']}]  реплик={row['recommended_replicas']}")


def _check_mape(model, builder: FeatureBuilder, df: pd.DataFrame) -> None:
    """Быстрая оценка качества на хвосте датасета."""
    if len(df) < 50:
        return
    recent = df.iloc[-50:]
    X, y = builder.get_X_y(recent)
    if len(X) < 10:
        return
    preds = predict_xgboost(model, X)
    from evaluation.metrics import safe_mape
    mape = safe_mape(y.values, preds)
    status = "OK" if mape < config.MAPE_RETRAIN_THRESHOLD else "ДЕГРАДАЦИЯ — рекомендуется переобучение"
    print(f"  [quality check] MAPE={mape:.1f}%  {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Интеграция с Prometheus (grafana-deploy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["train", "forecast", "monitor"],
        help="Режим запуска",
    )
    parser.add_argument(
        "--days", type=int, default=config.DEFAULT_DAYS_AGO,
        help=f"Глубина истории в днях (по умолчанию {config.DEFAULT_DAYS_AGO})",
    )
    parser.add_argument(
        "--step", default=config.DEFAULT_STEP,
        help=f"Шаг данных (по умолчанию {config.DEFAULT_STEP})",
    )
    parser.add_argument(
        "--no-neural-prophet", action="store_true",
        help="Пропустить обучение NeuralProphet (быстрее)",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Сохранить графики в saved_models/",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Интервал опроса в режиме monitor (сек, по умолчанию 60)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    dispatch = {
        "train":    mode_train,
        "forecast": mode_forecast,
        "monitor":  mode_monitor,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
