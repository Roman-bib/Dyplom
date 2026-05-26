"""
Система прогнозирования пиковых нагрузок (ВКР).

Запуск:
  python main.py --path ../data/azure_public.csv
  python main.py --path ../data/azure_public.csv --fast   # без NeuralProphet
"""

import argparse
import os
import sys
import warnings

import numpy as np
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

import config
from preprocessing.feature_engineering import FeatureBuilder, split_train_val_test
from preprocessing.data_cleaning import TimeSeriesCleaner
from models.forecasters import (
    train_xgboost, train_xgboost_random_search, predict_xgboost, predict_xgboost_wf,
    get_confidence_interval, feature_importance,
)
from models.multistep import recursive_forecast
from models.comparison import ModelComparison
from evaluation.metrics import (
    plot_forecast, plot_all_forecasts,
    export_metrics_csv, export_predictions_csv, export_feature_importance_csv,
)
from evaluation.peak_detection import PeakDetector, ResidualAnomalyDetector, plot_peaks
from evaluation.walk_forward import run_walk_forward
from retraining.scheduler import make_multi_model_train_fn
from retraining.drift_detector import ADWINDriftDetector


def main():
    parser = argparse.ArgumentParser(
        description="Система прогнозирования пиковых нагрузок (ВКР)",
    )
    parser.add_argument("--path", default=None,
                        help="Путь к CSV-файлу датасета")
    parser.add_argument("--fast", action="store_true",
                        help="Пропустить NeuralProphet (ускоряет запуск)")
    args = parser.parse_args()

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Загрузка данных
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    if args.path:
        from data_collection.csv_loader import load_csv
        df = load_csv(args.path)
    else:
        from data_collection.csv_loader import load_web_traffic
        df = load_web_traffic()

    n = len(df)
    print(f"Загружено {n} точек | "
          f"RPS: min={df['y'].min():.0f}  max={df['y'].max():.0f}  "
          f"mean={df['y'].mean():.0f}")
    print(f"Период: {df['ds'].iloc[0]}  ->  {df['ds'].iloc[-1]}\n")

    if n < 200:
        print("Недостаточно данных. Проверьте файл.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Предобработка и разбиение
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ШАГ 2: ПРЕДОБРАБОТКА")
    print("=" * 60)

    # Для 1-минутных данных 480 точек = 8 часов — слишком мало.
    # Берём ~10% датасета, но не менее 2 дней (2880 точек) и не более 14 дней (20160).
    TEST_H = max(2880, min(20160, n * 15 // 100))
    train_raw, val_raw, test_raw = split_train_val_test(df, test_hours=TEST_H, val_hours=TEST_H)

    cleaner = TimeSeriesCleaner()
    train, train_stats = cleaner.fit(train_raw).transform(train_raw)
    val,   _           = cleaner.transform(val_raw)
    test,  test_stats  = cleaner.transform(test_raw)
    cleaner.save(config.MODEL_SAVE_DIR)
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)} точек")
    print(f"  Очистка train: шум удалён={train_stats['n_outliers_removed']}, "
          f"легитимных пиков сохранено={train_stats['n_peaks_preserved']}")
    print(f"  Очистка test:  шум удалён={test_stats['n_outliers_removed']}, "
          f"легитимных пиков сохранено={test_stats['n_peaks_preserved']}\n")

    # ------------------------------------------------------------------
    # 3. Сравнение моделей (XGBoost, NeuralProphet, LSTM)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ШАГ 3: СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    comparator = ModelComparison(model_save_dir=config.MODEL_SAVE_DIR)
    comparator.run(train, val, test, include_prophet=not args.fast)
    comparator.save_best()

    plot_all_forecasts(train, val, test, predictions_dict=comparator.predictions_,
                       save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_full.png"),
                       zoom=False)
    plot_all_forecasts(train, val, test, predictions_dict=comparator.predictions_,
                       save_path=os.path.join(config.MODEL_SAVE_DIR, "comparison_zoom.png"),
                       zoom=True)

    # ------------------------------------------------------------------
    # 4. XGBoost: доверительный интервал + важность признаков
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ШАГ 4: ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ И ВАЖНОСТЬ ПРИЗНАКОВ")
    print("=" * 60)

    builder = comparator._builder  # тот же builder что обучал модели
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    model = comparator.models_.get("XGBoost") or \
            joblib.load(os.path.join(config.MODEL_SAVE_DIR, "xgboost.pkl"))
    preds = predict_xgboost(model, X_test)

    lower, upper = get_confidence_interval(
        X_train, y_train, X_val, y_val, X_test,
        save_dir=config.MODEL_SAVE_DIR,
    )
    imp_df = feature_importance(model, feature_names=list(X_train.columns))
    print("\nТоп-5 важных признаков:")
    print(imp_df.head(5).to_string(index=False))

    plot_forecast(train, val, test, preds, "XGBoost", lower=lower, upper=upper,
                  save_path=os.path.join(config.MODEL_SAVE_DIR, "forecast_xgboost.png"))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 5→6. Детекция пиков (IF + пороговый классификатор)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ШАГ 6: ДЕТЕКЦИЯ ПИКОВ")
    print("=" * 60)

    # Изолирующий лес обучается на остатке обучающей выборки
    anomaly_det = ResidualAnomalyDetector(
        contamination=getattr(config, "IF_CONTAMINATION", 0.05),
    )
    anomaly_det.fit(train_stats["residual"].values)
    anomaly_det.save(config.MODEL_SAVE_DIR)
    print(f"  IF обучен на {len(train_stats['residual'])} остатках")

    detector = PeakDetector(
        target_rps_per_replica=float(train["y"].max()) / config.MAX_REPLICAS,
        min_replicas=config.MIN_REPLICAS,
        max_replicas=config.MAX_REPLICAS,
    )
    detector.fit(train["y"])
    predicted_series = pd.Series(preds, index=y_test.index)
    _recompute = getattr(config, "PEAK_RECOMPUTE_EVERY", None)
    events_df = detector.detect_series(y_test, predicted_series, recompute_every=_recompute)
    summary = detector.summary(events_df)

    # IF классифицирует природу пика и корректирует число реплик
    test_residual = test_stats["residual"].reindex(
        pd.DatetimeIndex(events_df["timestamp"])
    ).fillna(0).values
    events_df["is_anomaly"] = anomaly_det.predict(test_residual)

    # Для аномальных точек пересчитываем реплики с коэффициентом запаса
    k_safety = getattr(config, "IF_SAFETY_FACTOR", 1.2)
    anomaly_mask = events_df["is_anomaly"]
    if anomaly_mask.any():
        events_df.loc[anomaly_mask, "recommended_replicas"] = (
            events_df.loc[anomaly_mask, "predicted"]
            .apply(lambda pred: detector._calculate_replicas(pred * k_safety))
        )

    print(f"  Метод: adaptive_percentile  порог={summary['threshold']:.0f} RPS")
    print(f"  Пиков: {summary['peaks_detected']} / {summary['total_points']} "
          f"({summary['peak_ratio_pct']}%)")
    print(f"  Уровни severity: {summary['severity_counts']}")
    n_anomaly = int(events_df["is_anomaly"].sum())
    print(f"  Аномальных пиков (IF): {n_anomaly}")

    if events_df["is_peak"].any():
        print(f"\n  Первые 5 пиков:")
        for _, row in events_df[events_df["is_peak"]].head(5).iterrows():
            anomaly_tag = " [ANOMALY]" if row.get("is_anomaly") else ""
            print(f"    {row['timestamp']}  факт={row['rps']:.0f}  "
                  f"прогноз={row['predicted']:.0f}  "
                  f"[{row['severity']}]{anomaly_tag}  реплик={row['recommended_replicas']}")

    plot_peaks(events_df, summary,
               save_path=os.path.join(config.MODEL_SAVE_DIR, "peaks.png"),
               max_replicas=config.MAX_REPLICAS)

    # ------------------------------------------------------------------
    # 6b. Многошаговый прогноз (recursive, горизонт H из config)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ШАГ 6b: МНОГОШАГОВЫЙ ПРОГНОЗ")
    print("=" * 60)

    _horizons = getattr(config, "FORECAST_HORIZONS_PERIODS", [3, 6, 12])
    H = _horizons[1]  # 30 мин при шаге 5 мин
    history_ms = pd.concat([train, val, test]).sort_values("ds").reset_index(drop=True)
    ms_forecast = recursive_forecast(
        history_df=history_ms,
        model=model,
        predict_fn=predict_xgboost,
        builder=builder,
        horizon=H,
    )
    print(f"  Горизонт: H={H} шагов вперёд от конца тест-периода")
    for _, r in ms_forecast.iterrows():
        print(f"    {r['ds']}  ŷ={r['y_hat']:.1f} RPS")

    ms_csv = os.path.join(config.MODEL_SAVE_DIR, "multistep_forecast.csv")
    ms_forecast.to_csv(ms_csv, index=False)

    # График: хвост теста + многошаговый прогноз
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _tail = test.tail(max(H * 4, 60))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_tail["ds"], _tail["y"], color="steelblue", linewidth=1.5, label="Факт (тест)")
    ax.plot(ms_forecast["ds"], ms_forecast["y_hat"],
            color="crimson", linewidth=1.8, linestyle="--",
            marker="o", markersize=4, label=f"Прогноз H={H} шагов")
    ax.axvline(test["ds"].iloc[-1], color="gray", linestyle=":", linewidth=1.0,
               label="Граница известных данных")
    ax.set_xlabel("Время")
    ax.set_ylabel("RPS")
    ax.set_title(f"Рекурсивный многошаговый прогноз (H={H} шагов вперёд)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _ms_path = os.path.join(config.MODEL_SAVE_DIR, "multistep_forecast.png")
    plt.savefig(_ms_path, dpi=150)
    plt.close()
    print(f"  Сохранено: {_ms_path}")

    # ------------------------------------------------------------------
    # 7. Walk-forward валидация с адаптивным переобучением
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ШАГ 7: WALK-FORWARD ВАЛИДАЦИЯ")
    print("=" * 60)

    drift = ADWINDriftDetector(
        delta=getattr(config, "ADWIN_DELTA", 0.002),
        min_obs=getattr(config, "ADWIN_MIN_OBS", 30),
        cooldown_n=getattr(config, "ADWIN_COOLDOWN_N", 20),
        n_fresh=getattr(config, "ADWIN_N_FRESH", 0),
        confirmation_n=getattr(config, "ADWIN_CONFIRMATION_N", 10),
    )
    wf_model, wf_best_params, _ = train_xgboost_random_search(
        X_train, y_train, X_val, y_val,
        save_path=os.path.join(config.MODEL_SAVE_DIR, "xgboost_wf.pkl"),
    )

    # Гиперпараметры Prophet из шага сравнения (для refit без повторного grid-search)
    prophet_params = None
    if "Prophet" in comparator.models_:
        prophet_params = getattr(comparator.models_["Prophet"], "_best_params", None)

    wf_result = run_walk_forward(
        train=train, val=val, test=test,
        initial_model=wf_model,
        predict_fn=predict_xgboost_wf,
        train_fn=make_multi_model_train_fn(
            builder,
            save_dir=config.MODEL_SAVE_DIR,
            xgb_params=wf_best_params,
            prophet_best_params=prophet_params,
            include_lstm=not args.fast,
            include_prophet=not args.fast,
        ),
        builder=builder,
        drift_detector=drift,
        save_dir=config.MODEL_SAVE_DIR,
        cleaner=cleaner,
    )
    wf_s = wf_result["summary"]
    print(f"  MAE адаптивная:   {wf_s['mae_adaptive']}")
    print(f"  MAE фиксированная:{wf_s['mae_baseline']}")
    print(f"  Улучшение:        {wf_s['improvement_pct']}%")
    print(f"  Переобучений:     {wf_s['n_retrains']}")

    # --- График: все статичные модели + адаптивная vs факт ---
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    wf_df = wf_result["results_df"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test["ds"], test["y"], color="#27ae60", linewidth=2,
            label="Факт (тест)", zorder=5)
    _colors = ["#e74c3c", "#9b59b6", "#e67e22"]
    for (name, preds), col in zip(comparator.predictions_.items(), _colors):
        ax.plot(test["ds"].values[:len(preds)],
                np.clip(preds[:len(test)], 0, None),
                color=col, linestyle="--", linewidth=1.2,
                label=f"{name} (статичная)", alpha=0.8)
    ax.plot(pd.to_datetime(wf_df["timestamp"]), wf_df["y_pred"],
            color="steelblue", linewidth=1.5,
            label="Адаптивная (walk-forward)", zorder=4)
    for ts in wf_s["retrain_timestamps"]:
        ax.axvline(ts, color="crimson", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Время")
    ax.set_ylabel("RPS")
    ax.set_title("Сравнение моделей и адаптивной системы на тест-периоде")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _cmp_path = os.path.join(config.MODEL_SAVE_DIR, "comparison_vs_adaptive.png")
    plt.savefig(_cmp_path, dpi=150)
    plt.close()
    print(f"  Сохранено: {_cmp_path}")

    # ------------------------------------------------------------------
    # 8. Экспорт результатов
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ШАГ 8: ЭКСПОРТ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    export_metrics_csv(comparator.results_, config.MODEL_SAVE_DIR)
    export_predictions_csv(test, comparator.predictions_, lower, upper, config.MODEL_SAVE_DIR)
    export_feature_importance_csv(imp_df, config.MODEL_SAVE_DIR)

    print(f"\nВсё сохранено в {config.MODEL_SAVE_DIR}/")
    print(f"  comparison_full.png      — сравнение всех моделей")
    print(f"  forecast_xgboost.png     — прогноз XGBoost с CI")
    print(f"  peaks.png                — детекция пиков")
    print(f"  multistep_forecast.png   — многошаговый прогноз (H шагов вперёд)")
    print(f"  multistep_forecast.csv   — значения многошагового прогноза")
    print(f"  walk_forward_mae.png     — walk-forward валидация")
    print(f"  metrics_summary.csv      — метрики всех моделей")
    print(f"  predictions.csv          — прогнозы на тестовом периоде")
    print(f"  feature_importance.csv   — важность признаков")


if __name__ == "__main__":
    main()
