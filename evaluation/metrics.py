"""
Метрики оценки качества прогнозирования (расширенный набор).

Базовые (глава 1.3.2 ВКР):
  MAE   = (1/n) * Σ|y_i - ŷ_i|
  RMSE  = √((1/n) * Σ(y_i - ŷ_i)²)
  MAPE  = (1/n) * Σ|y_i - ŷ_i| / |y_i| * 100%       (формула 1.3)

Расширенный набор (добавлены после аудита, см. AUDIT_REPORT.md, шаг 8):
  SMAPE = (200/n) * Σ |y - ŷ| / (|y| + |ŷ|)         — устойчивый аналог MAPE
  MASE  = MAE / MAE(seasonal_naive)                  — нормированная ошибка
  peak_focused_mae  — MAE только на точках, где факт ≥ peak_threshold
  peak_detection_metrics  — precision/recall/F1/MCC для пик/не-пик
  coverage_metrics        — покрытие и ширина CI для квантильных моделей
  lead_time_metric        — упредительность прогноза пиков

Для пик-детекции это критично: модель с маленьким MAE на хвосте может
проваливаться именно на пиках, которые мы и хотим ловить. Стандартный MAE
этого не покажет.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, matthews_corrcoef,
)
import matplotlib.pyplot as plt


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    MAPE, устойчивая к нулям.
    Точки, где |y_true| < eps, исключаются из расчёта.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Symmetric MAPE.  Лучше MAPE, потому что:
      - симметрична относительно under-/over-prediction;
      - устойчива к малым y_true (в знаменателе и y_true, и y_pred);
      - всегда лежит в [0, 200%].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(model) / MAE(seasonal_naive on train).
    < 1  → модель лучше seasonal-naive baseline
    = 1  → ровно как seasonal-naive
    > 1  → хуже baseline (модель бесполезна)

    season=1  — простой naive (повторение последнего значения)
    season=24 — суточная периодика для часовых данных
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= season:
        return float("nan")
    naive_errors = np.abs(y_train[season:] - y_train[:-season])
    scale = float(np.mean(naive_errors))
    if scale < 1e-9:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred) / scale)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R².

    R² = 1 - SS_res / SS_tot
    R² = 1.0  → идеальное предсказание
    R² = 0.0  → модель не лучше среднего значения y_true
    R² < 0    → модель ХУЖЕ предсказания «всегда среднее»

    Это самая надёжная scale-free метрика для оценки регрессии:
    она не зависит от диапазона y и одинаково интерпретируется для
    rps в тысячах и cpu_usage в процентах.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def is_mape_reliable(y_true: np.ndarray, near_zero_threshold: float = 0.05) -> bool:
    """
    MAPE надёжна только если в данных нет близких к нулю значений.

    Критерий: 10-й перцентиль |y| должен быть > near_zero_threshold * mean(|y|).
    Иначе деление на маленькие y взрывает MAPE до сотен процентов.
    """
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) == 0:
        return False
    abs_y = np.abs(y_true)
    mean_y = float(np.mean(abs_y))
    if mean_y < 1e-9:
        return False
    p10 = float(np.percentile(abs_y, 10))
    return p10 > near_zero_threshold * mean_y


def evaluate(
    y_true,
    y_pred,
    model_name: str = "Model",
    verbose: bool = True,
    y_train: Optional[np.ndarray] = None,
    season: int = 1,
) -> Dict[str, float]:
    """
    Вычисляет MAE, RMSE, MAPE, SMAPE, R², и (если задан y_train) MASE.

    Также возвращает MAPE_reliable — True если MAPE можно доверять
    (нет близких к нулю значений). Если False — следует смотреть на
    SMAPE и R² вместо MAPE.

    Returns
    -------
    dict с ключами:
      "MAE", "RMSE", "MAPE", "SMAPE", "R2", "MAPE_reliable",
      опционально "MASE"
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape  = safe_mape(y_true, y_pred)
    smape = safe_smape(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape_ok = is_mape_reliable(y_true)

    out: Dict[str, float] = {
        "MAE": mae, "RMSE": rmse,
        "MAPE": mape, "SMAPE": smape, "R2": r2,
        "MAPE_reliable": mape_ok,
    }

    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, season=season)

    if verbose:
        mape_str = f"{mape:6.2f}%" if mape_ok else f"({mape:6.2f}%)*"
        msg = (f"  {model_name:<20} MAE={mae:8.2f}  RMSE={rmse:8.2f}  "
               f"MAPE={mape_str}  SMAPE={smape:6.2f}%  R2={r2:6.3f}")
        if "MASE" in out:
            msg += f"  MASE={out['MASE']:.3f}"
        if not mape_ok:
            msg += "  (* MAPE ненадёжна: y близко к 0)"
        print(msg)

    return out


# ---------------------------------------------------------------------------
# Метрики, специфичные для задачи пик-детекции
# ---------------------------------------------------------------------------

def peak_focused_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
) -> float:
    """
    MAE на точках, где фактическое y ≥ peak_threshold.

    Зачем: общий MAE может быть мал из-за «тихих» периодов, при этом
    модель плохо угадывает пики. Эта метрика отвечает на вопрос:
    «насколько точно модель предсказывает именно пиковые точки?»
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= peak_threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def peak_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
) -> Dict[str, float]:
    """
    Бинарная классификация «пик / не-пик» по порогу peak_threshold.

      precision = TP / (TP + FP)   — насколько мы НЕ ложно-тревожим
      recall    = TP / (TP + FN)   — насколько мы ловим реальные пики
      F1        — гармоническое среднее
      MCC       — Matthews Correlation Coefficient (устойчив к дисбалансу
                  классов: пиков обычно намного меньше чем не-пиков)
    """
    y_true_bin = (np.asarray(y_true, dtype=float) >= peak_threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred, dtype=float) >= peak_threshold).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

    # Если в выборке нет ни одного пика — precision/recall/F1 не определены
    if y_true_bin.sum() == 0 and y_pred_bin.sum() == 0:
        prec = rec = f1 = mcc = float("nan")
    else:
        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec  = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1   = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        try:
            mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))
        except ValueError:
            mcc = float("nan")

    return {
        "peak_precision": prec,
        "peak_recall":    rec,
        "peak_f1":        f1,
        "peak_mcc":       mcc,
        "peak_TP": tp, "peak_FP": fp, "peak_FN": fn, "peak_TN": tn,
    }


def coverage_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.8,
) -> Dict[str, float]:
    """
    Метрики качества квантильного прогноза (доверительного интервала).

      coverage  = доля точек, где y_true ∈ [lower, upper]
      width     = средняя ширина интервала (в единицах y)
      width_n   = ширина / std(y_true) — нормированная ширина

    Идеал: coverage ≈ nominal_coverage при минимальной width.
    Например, для 80% CI ожидаем coverage близко к 0.80;
    coverage намного меньше → интервал слишком узкий (под-уверенность);
    coverage намного больше → слишком широкий (бесполезен для алертинга).
    """
    y_true = np.asarray(y_true, dtype=float)
    lower  = np.asarray(lower,  dtype=float)
    upper  = np.asarray(upper,  dtype=float)
    if len(y_true) == 0:
        return {"coverage": float("nan"), "width": float("nan"),
                "width_n": float("nan"), "nominal": nominal_coverage}

    inside = (y_true >= lower) & (y_true <= upper)
    coverage = float(inside.mean())
    width = float(np.mean(upper - lower))
    sigma = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 1.0
    width_n = width / sigma if sigma > 0 else float("nan")

    return {
        "coverage":   coverage,
        "width":      width,
        "width_n":    width_n,
        "nominal":    nominal_coverage,
        "miscoverage_abs": abs(coverage - nominal_coverage),
    }


def lead_time_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
    timestamps: Optional[Sequence] = None,
    step_minutes: float = 60.0,
) -> Dict[str, float]:
    """
    Lead time — насколько ЗАРАНЕЕ модель предсказывает пик.

    Алгоритм: для каждого реального пика в y_true ищется ближайший
    предшествующий момент времени, в который y_pred уже преодолел порог.
    Lead time = разница в минутах. Если y_pred не пересёк порог до пика —
    этот пик считается «пропущенным» (не учитывается в среднем lead-time,
    но учитывается в miss_rate).

    Returns
    -------
    {
        "mean_lead_min": среднее упреждение в минутах,
        "median_lead_min": медианное,
        "miss_rate": доля пиков, не предсказанных вовремя,
        "peaks_total": общее число пиков в y_true,
    }
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    peak_idx = np.where(y_true >= peak_threshold)[0]
    if len(peak_idx) == 0:
        return {
            "mean_lead_min": float("nan"),
            "median_lead_min": float("nan"),
            "miss_rate": float("nan"),
            "peaks_total": 0,
        }

    # Для каждого пика: ищем ближайший прошлый i, где y_pred[i] ≥ threshold
    lead_steps = []
    misses = 0
    pred_above = y_pred >= peak_threshold

    for p in peak_idx:
        # Идём назад от p, ищем первое True
        prior = pred_above[:p + 1]
        if not prior.any():
            misses += 1
            continue
        # Lead = p - индекс пересечения порога
        first_alarm = int(np.argmax(prior))
        # argmax возвращает первый True слева; нам нужен ближайший к p
        # → ищем последний True в prior
        last_alarm_idx = np.where(prior)[0][-1]
        lead_steps.append(p - last_alarm_idx)

    mean_lead_min = (
        float(np.mean(lead_steps) * step_minutes) if lead_steps else float("nan")
    )
    median_lead_min = (
        float(np.median(lead_steps) * step_minutes) if lead_steps else float("nan")
    )

    return {
        "mean_lead_min":   mean_lead_min,
        "median_lead_min": median_lead_min,
        "miss_rate":       float(misses / len(peak_idx)),
        "peaks_total":     int(len(peak_idx)),
    }


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Выводит итоговую таблицу сравнения моделей.

    Включает базовые (MAE/RMSE/MAPE/SMAPE) и пик-метрики (F1) если доступны.
    Победитель помечается по MAE — это интерпретируемая базовая метрика.
    """
    if not results:
        return

    winner = min(results, key=lambda n: results[n].get("MAE", float("inf")))

    has_f1 = any("peak_f1" in m for m in results.values())
    has_mase = any("MASE" in m for m in results.values())

    columns = ["MAE", "RMSE", "MAPE%", "SMAPE%"]
    if has_mase:
        columns.append("MASE")
    if has_f1:
        columns.append("PeakF1")
    columns.append("Время,с")

    header = f"\n{'Модель':<22} " + " ".join(f"{c:>9}" for c in columns)
    sep = "-" * (22 + 10 * len(columns))
    print(sep)
    print(header)
    print(sep)

    for name, m in results.items():
        mark = " [*]" if name == winner else "    "
        cells = [
            f"{m['MAE']:>9.2f}",
            f"{m['RMSE']:>9.2f}",
            f"{m['MAPE']:>8.2f}%",
            f"{m.get('SMAPE', float('nan')):>8.2f}%",
        ]
        if has_mase:
            v = m.get("MASE")
            cells.append(f"{v:>9.3f}" if v is not None and not np.isnan(v) else f"{'-':>9}")
        if has_f1:
            v = m.get("peak_f1")
            cells.append(f"{v:>9.3f}" if v is not None and not np.isnan(v) else f"{'-':>9}")

        t = m.get("train_time_s", "-")
        cells.append(f"{t:>9.2f}" if isinstance(t, (int, float)) else f"{t:>9}")

        print(f"  {name+mark:<24} " + " ".join(cells))

    print(sep)
    print(f"  [*] - best model by MAE\n")


def plot_forecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    zoom: bool = False,
) -> None:
    """
    Один график: полный ряд (zoom=False) или только тест-период (zoom=True).
    """
    predictions = np.clip(predictions, 0, None)
    if lower is not None:
        lower = np.clip(lower, 0, None)
    if upper is not None:
        upper = np.clip(upper, 0, None)

    fig, ax = plt.subplots(figsize=(14, 5))

    if zoom:
        n = len(predictions)
        test_z = test.iloc[:n]
        ax.plot(test_z["ds"], test_z["y"], color="#27ae60", label="Факт",
                linewidth=2, zorder=5)
        ax.plot(test_z["ds"], predictions, color="#e74c3c",
                label=f"{model_name} (прогноз)", linewidth=1.8, linestyle="--")
        if lower is not None and upper is not None:
            ax.fill_between(test_z["ds"], lower[:n], upper[:n],
                            alpha=0.2, color="#e74c3c",
                            label="Доверительный интервал 80%")
        ax.set_title(f"{model_name}: тест-период", fontsize=13)
    else:
        ax.plot(train["ds"], train["y"], color="steelblue", label="Train",
                linewidth=0.8, alpha=0.6)
        ax.plot(val["ds"], val["y"], color="orange", label="Validation",
                linewidth=0.8, alpha=0.8)
        ax.plot(test["ds"], predictions, color="#e74c3c",
                label=f"{model_name} (прогноз)", linewidth=1.5, linestyle="--", zorder=3)
        ax.plot(test["ds"], test["y"], color="#27ae60", label="Test (факт)",
                linewidth=1.5, zorder=4)
        if lower is not None and upper is not None:
            ax.fill_between(test["ds"], lower, upper,
                            alpha=0.2, color="#e74c3c",
                            label="Доверительный интервал 80%")
        ax.axvspan(test["ds"].iloc[0], test["ds"].iloc[-1],
                   alpha=0.07, color="green")
        ax.set_title(f"{model_name}: полный временной ряд", fontsize=13)

    ax.set_xlabel("Время")
    ax.set_ylabel("RPS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_forecasts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    zoom: bool = False,
) -> None:
    """
    Один график: полный ряд (zoom=False) или только тест-период (zoom=True).
    """
    colors = ["#e74c3c", "#9b59b6", "#e67e22", "#1abc9c", "#c0392b"]
    test_dates = test["ds"].values
    y_actual = test["y"].values

    fig, ax = plt.subplots(figsize=(16, 6))

    if zoom:
        ax.plot(test_dates, y_actual, color="#27ae60", label="Факт",
                linewidth=2, zorder=5)
        for (name, preds), color in zip(predictions_dict.items(), colors):
            clipped = np.clip(preds[:len(y_actual)], 0, None)
            ax.plot(test_dates[:len(clipped)], clipped,
                    color=color, label=name, linewidth=1.5, linestyle="--", alpha=0.9)
        ax.set_title("Сравнение моделей: тест-период", fontsize=13)
    else:
        ax.plot(train["ds"], train["y"], color="steelblue", label="Train",
                linewidth=0.8, alpha=0.6)
        ax.plot(val["ds"], val["y"], color="orange", label="Validation",
                linewidth=0.8, alpha=0.8)
        ax.plot(test["ds"], test["y"], color="#27ae60", label="Test (факт)",
                linewidth=1.5)
        for (name, preds), color in zip(predictions_dict.items(), colors):
            clipped = np.clip(preds[:len(y_actual)], 0, None)
            ax.plot(test_dates[:len(clipped)], clipped,
                    color=color, label=name, linewidth=1.5, linestyle="--", alpha=0.9)
        ax.axvspan(test["ds"].iloc[0], test["ds"].iloc[-1],
                   alpha=0.07, color="green")
        ax.set_title("Сравнение моделей: полный временной ряд", fontsize=13)

    ax.set_xlabel("Время")
    ax.set_ylabel("RPS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV-экспорт результатов
# ---------------------------------------------------------------------------

def export_metrics_csv(results: dict, save_dir: str) -> str:
    """Сохраняет сводную таблицу метрик всех моделей в CSV."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for model_name, m in results.items():
        row = {"model": model_name}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, "metrics_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_predictions_csv(
    test: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    save_dir: str,
) -> str:
    """Сохраняет прогнозы всех моделей + доверительный интервал XGBoost в CSV."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({"timestamp": test["ds"].values, "y_true": test["y"].values})
    for name, preds in predictions_dict.items():
        df[f"pred_{name}"] = np.clip(preds[: len(df)], 0, None)
    if lower is not None:
        df["ci_lower"] = lower[: len(df)]
    if upper is not None:
        df["ci_upper"] = upper[: len(df)]
    path = os.path.join(save_dir, "predictions.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_feature_importance_csv(imp_df: pd.DataFrame, save_dir: str) -> str:
    """Сохраняет таблицу важности признаков XGBoost в CSV."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "feature_importance.csv")
    imp_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
