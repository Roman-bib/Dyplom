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


def evaluate(
    y_true,
    y_pred,
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Вычисляет MAE, RMSE, MAPE для пары (факт, прогноз).

    Returns
    -------
    dict с ключами "MAE", "RMSE", "MAPE"
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = safe_mape(y_true, y_pred)

    if verbose:
        print(f"  {model_name:<20} MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Выводит итоговую таблицу сравнения моделей.

    Пример вывода:
    ┌────────────────────┬──────────┬──────────┬────────┬──────────────┐
    │ Модель             │      MAE │     RMSE │  MAPE% │  Время обуч. │
    ├────────────────────┼──────────┼──────────┼────────┼──────────────┤
    │ LinearRegression   │   120.34 │   180.21 │   8.45 │         0.12 │
    │ XGBoost        ★  │    45.12 │    67.89 │   3.21 │         1.45 │
    │ Prophet            │    78.90 │   110.34 │   5.67 │        42.30 │
    └────────────────────┴──────────┴──────────┴────────┴──────────────┘
    """
    if not results:
        return

    # Находим победителя по MAPE
    winner = min(results, key=lambda n: results[n].get("MAPE", float("inf")))

    header = f"\n{'Модель':<22} {'MAE':>9} {'RMSE':>9} {'MAPE%':>7} {'Время,с':>9}"
    sep = "-" * 62
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        mark = " [*]" if name == winner else "    "
        t = m.get("train_time_s", "-")
        t_str = f"{t:>9.2f}" if isinstance(t, (int, float)) else f"{t:>9}"
        print(
            f"  {name+mark:<24} {m['MAE']:>9.2f} {m['RMSE']:>9.2f}"
            f" {m['MAPE']:>6.2f}%{t_str}"
        )
    print(sep)
    print(f"  [*] - best model by MAPE\n")


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
        ax.plot(test["ds"], test["y"], color="#27ae60", label="Test (факт)",
                linewidth=1.5)
        ax.plot(test["ds"], predictions, color="#e74c3c",
                label=f"{model_name} (прогноз)", linewidth=1.5, linestyle="--")
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
