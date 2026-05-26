"""
Загрузчик данных из CSV-файла.

Читает CSV с временным рядом и приводит к стандартному формату {ds, y}.

Поддерживаемые форматы:
  - timestamp, rps, concurrent_users, ... (web_traffic.csv)
  - Любой CSV с одной datetime-колонкой и одной числовой
  - Автоопределение колонок если timestamp_col/value_col не указаны
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Путь к датасету из Code/ML.py (относительно proactive-scaler)
DEFAULT_CSV_PATH = Path(__file__).parent.parent.parent / "Code" / "data" / "web_traffic.csv"


def _detect_timestamp_col(df: pd.DataFrame) -> str:
    """Возвращает имя первой колонки, которую удаётся распарсить как datetime."""
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col])
            if parsed.notna().mean() > 0.9:
                return col
        except Exception:
            continue
    raise ValueError(
        f"Не удалось автоматически определить колонку с датой/временем.\n"
        f"Доступные колонки: {list(df.columns)}\n"
        f"Укажите явно: --timestamp-col <имя>"
    )


def _detect_value_col(df: pd.DataFrame, timestamp_col: str) -> str:
    """Возвращает имя первой числовой колонки, не являющейся timestamp."""
    numeric_cols = [
        col for col in df.columns
        if col != timestamp_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        raise ValueError(
            f"Не найдено числовых колонок (кроме '{timestamp_col}').\n"
            f"Доступные колонки: {list(df.columns)}\n"
            f"Укажите явно: --value-col <имя>"
        )
    return numeric_cols[0]


def load_csv(
    path: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    value_col: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Загружает CSV и возвращает DataFrame {ds, y}.

    Parameters
    ----------
    path          : путь к CSV (None = web_traffic.csv)
    timestamp_col : колонка с датой/временем (None = автоопределение)
    value_col     : целевая метрика (None = первая числовая колонка)
    start, end    : фильтр по дате ("2023-06-01" и т.д.)

    Returns
    -------
    DataFrame с колонками ds (datetime) и y (float), отсортированный по ds.
    """
    csv_path = Path(path) if path else DEFAULT_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV-файл не найден: {csv_path}\n"
            "Убедитесь что файл существует или укажите --path."
        )

    # Читаем без parse_dates — сначала автоопределим нужные колонки
    df = pd.read_csv(csv_path)

    ts_col = timestamp_col or _detect_timestamp_col(df)
    val_col = value_col or _detect_value_col(df, ts_col)

    if ts_col not in df.columns:
        raise ValueError(
            f"Колонка '{ts_col}' не найдена. Доступные: {list(df.columns)}"
        )
    if val_col not in df.columns:
        raise ValueError(
            f"Колонка '{val_col}' не найдена. Доступные: {list(df.columns)}"
        )

    print(f"  timestamp -> '{ts_col}',  value -> '{val_col}'")

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.rename(columns={ts_col: "ds", val_col: "y"})

    # Сохраняем экзогенные колонки если они есть в CSV
    try:
        import config as _cfg
        exog_cols = [c for c in getattr(_cfg, "EXOG_COLS", []) if c in df.columns]
    except ImportError:
        exog_cols = []
    keep_cols = ["ds", "y"] + exog_cols
    df = df[[c for c in keep_cols if c in df.columns]]
    if exog_cols:
        print(f"  Экзогенные признаки: {exog_cols}")

    # Фильтр по дате до очистки, чтобы не интерполировать ненужные участки
    if start:
        df = df[df["ds"] >= pd.to_datetime(start)]
    if end:
        df = df[df["ds"] <= pd.to_datetime(end)]

    # Полный пайплайн предобработки (Рисунок 2.2 ВКР):
    # сортировка, удаление дубликатов, расчёт медианного шага τ,
    # выравнивание на регулярную сетку, линейная интерполяция, float64
    from preprocessing.data_cleaning import clean_timeseries
    df, stats = clean_timeseries(df, ts_col="ds", value_col="y")
    if stats.get("n_gaps_filled", 0) > 0:
        print(f"  Заполнено разрывов: {stats['n_gaps_filled']} "
              f"(шаг {stats['step_minutes']:.1f} мин)")
    return df.reset_index(drop=True)


def load_web_traffic(
    months: int = 12,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Удобная обёртка: загружает web_traffic.csv за указанное число месяцев.

    Parameters
    ----------
    months : сколько месяцев взять начиная с start
    start  : начальная дата
    """
    end_dt = pd.to_datetime(start) + pd.DateOffset(months=months)
    return load_csv(
        timestamp_col="timestamp",
        value_col="rps",
        start=start,
        end=str(end_dt.date()),
    )


def describe_csv(path: Optional[str] = None) -> None:
    """Выводит статистику по датасету (удобно для проверки перед обучением)."""
    df = load_csv(path)
    n = len(df)
    step = (df["ds"].iloc[1] - df["ds"].iloc[0]) if n > 1 else None
    print(f"Файл:    {path or DEFAULT_CSV_PATH}")
    print(f"Строк:   {n}")
    print(f"Период:  {df['ds'].iloc[0]}  ->  {df['ds'].iloc[-1]}")
    print(f"Шаг:     {step}")
    print(f"RPS:     min={df['y'].min():.1f}  max={df['y'].max():.1f}  "
          f"mean={df['y'].mean():.1f}  std={df['y'].std():.1f}")


def load_wiki_pageviews(path: str, top_n_pages: int = 50) -> pd.DataFrame:
    """
    Загружает датасет Wikipedia Web Traffic (Kaggle wide-format).

    Формат входного файла:
      - Первая колонка: Page (название страницы)
      - Остальные колонки: даты в формате YYYYMMDDNN (напр. 2018010100)
      - Значения: количество просмотров страницы в этот день

    Выход: DataFrame {ds (datetime), y (float)} — суммарные просмотры
    по top_n_pages самых популярных страниц за каждый день.

    Parameters
    ----------
    path      : путь к CSV-файлу
    top_n_pages : сколько страниц суммировать (None = все)
    """
    df_raw = pd.read_csv(path, index_col=0)

    # Отбираем наиболее популярные страницы для стабильного сигнала
    if top_n_pages is not None and top_n_pages < len(df_raw):
        total_views = df_raw.sum(axis=1)
        df_raw = df_raw.loc[total_views.nlargest(top_n_pages).index]

    # Суммируем просмотры по всем страницам за каждый день
    daily_views = df_raw.sum(axis=0)

    # Парсим даты: YYYYMMDDNN → datetime (отбрасываем последние 2 символа NN)
    dates = []
    values = []
    for col_name, val in daily_views.items():
        date_str = str(col_name)[:8]   # YYYYMMDD
        try:
            dt = pd.to_datetime(date_str, format="%Y%m%d")
            dates.append(dt)
            values.append(float(val))
        except ValueError:
            continue

    result = pd.DataFrame({"ds": dates, "y": values})
    result = result.sort_values("ds").reset_index(drop=True)
    result["y"] = result["y"].fillna(0.0)

    print(f"  Wikipedia: {len(result)} дней, "
          f"суммарные просмотры: min={result['y'].min():.0f}, "
          f"max={result['y'].max():.0f}, mean={result['y'].mean():.0f}")
    return result


def load_faas_dataset(top_n_instances: int = None) -> pd.DataFrame:
    """
    Загружает ByteDance/CloudTimeSeriesData (FaaS) с HuggingFace.

    Формат: date | data | cols (instance_N)
    Шаг: 10 минут, период: 2022-04-02 — 2024-10-24, 1113 инстансов.

    Суммирует нагрузку по всем (или top_n_instances) инстансам за каждый
    момент времени → суммарная нагрузка FaaS-платформы.

    Parameters
    ----------
    top_n_instances : если задан — берёт только N самых нагруженных инстансов
    """
    from datasets import load_dataset

    print("Загрузка ByteDance/CloudTimeSeriesData (FaaS)...")
    ds = load_dataset("ByteDance/CloudTimeSeriesData", split="train")
    df = ds.to_pandas()

    if top_n_instances is not None:
        top = (
            df.groupby("cols")["data"]
            .sum()
            .nlargest(top_n_instances)
            .index
        )
        df = df[df["cols"].isin(top)]

    df["date"] = pd.to_datetime(df["date"])
    df_agg = (
        df.groupby("date")["data"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "data": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    df_agg["y"] = df_agg["y"].clip(lower=0).fillna(0.0)

    print(f"  FaaS: {len(df_agg)} точек | шаг 10 мин | "
          f"y: min={df_agg['y'].min():.0f}, max={df_agg['y'].max():.0f}, "
          f"mean={df_agg['y'].mean():.0f}")
    print(f"  Период: {df_agg['ds'].iloc[0]} -> {df_agg['ds'].iloc[-1]}")
    return df_agg


if __name__ == "__main__":
    describe_csv()
