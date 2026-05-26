"""
Коннектор к Prometheus HTTP API.

Адаптирован под стек grafana-deploy (github.com/artemonsh/grafana-deploy):
  - Бэкенд: Litestar на порту 8080
  - Метрика: http_requests_total{method, path, status_code}
  - Scrape interval: 3s

Ключевые нюансы:
  - http_requests_total возвращает несколько временных рядов (по комбинации
    меток method/path/status_code). Используем sum() для агрегации в один ряд.
  - Рекомендованный запрос: sum(rate(http_requests_total[5m]))
    — это RPS (запросов в секунду), усреднённый за 5-минутное окно.
"""

import datetime
import time
from typing import Optional

import pandas as pd


def _connect(url: str, token: str = ""):
    """Создаёт PrometheusConnect с опциональным Bearer-токеном."""
    from prometheus_api_client import PrometheusConnect

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return PrometheusConnect(url=url, disable_ssl=True, headers=headers)


def check_connection(prometheus_url: str, token: str = "") -> bool:
    """
    Проверяет доступность Prometheus.
    Возвращает True если сервер отвечает, False иначе.
    """
    try:
        prom = _connect(prometheus_url, token)
        prom.custom_query("up")
        return True
    except Exception:
        return False


def fetch_metric(
    query: str,
    days_ago: int = 7,
    step: str = "5min",
    prometheus_url: str = "http://localhost:9090",
    token: str = "",
    end_time: Optional[datetime.datetime] = None,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """
    Запрашивает метрику из Prometheus (range query) и возвращает DataFrame.

    Parameters
    ----------
    query          : PromQL-запрос. Должен возвращать ровно один временной ряд.
                     Для http_requests_total используйте агрегацию:
                     "sum(rate(http_requests_total[5m]))"
    days_ago       : глубина исторических данных (дней)
    step           : шаг между точками ("5min", "1min", "1h" и т.д.)
    prometheus_url : адрес Prometheus
    token          : Bearer-токен (если требуется)
    end_time       : конец диапазона (None = сейчас)
    retries        : число попыток при ошибках сети
    retry_delay    : пауза между попытками (сек)

    Returns
    -------
    DataFrame с колонками:
      ds — datetime (UTC)
      y  — float (RPS или иная метрика)

    Raises
    ------
    RuntimeError   : Prometheus недоступен или запрос вернул пустой результат
    """
    prom = _connect(prometheus_url, token)

    if end_time is None:
        end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days_ago)

    # Prometheus duration format: "5m" not "5min"
    if step.endswith("min"):
        prom_step = step[:-3] + "m"
    elif step.endswith("hour"):
        prom_step = step[:-4] + "h"
    else:
        prom_step = step

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            result = prom.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=prom_step,
            )
            break
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                print(f"  Prometheus: попытка {attempt}/{retries} не удалась: {exc}")
                time.sleep(retry_delay)
    else:
        raise RuntimeError(
            f"Не удалось получить данные из Prometheus ({prometheus_url}) "
            f"после {retries} попыток. Последняя ошибка: {last_err}"
        )

    if not result:
        raise RuntimeError(
            f"Prometheus вернул пустой результат для запроса:\n  {query}\n"
            "Убедитесь что бэкенд запущен и Locust генерирует трафик."
        )

    if len(result) > 1:
        # Несколько временных рядов — скорее всего запрос без sum()
        print(
            f"  ВНИМАНИЕ: запрос вернул {len(result)} серий вместо одной. "
            "Используется sum агрегация. Рекомендуется добавить sum() в PromQL."
        )
        df = _aggregate_multiple_series(result)
    else:
        raw_values = result[0]["values"]
        df = pd.DataFrame(raw_values, columns=["ds", "y"])

    df["ds"] = pd.to_datetime(df["ds"], unit="s", utc=True).dt.tz_localize(None)
    df["y"] = df["y"].astype("float64")
    df = df.sort_values("ds").reset_index(drop=True)

    return df


def _aggregate_multiple_series(result: list) -> pd.DataFrame:
    """
    Суммирует несколько временных рядов по timestamp.
    Применяется когда запрос вернул несколько label-комбинаций.
    """
    frames = []
    for series in result:
        tmp = pd.DataFrame(series["values"], columns=["ds", "y"])
        tmp["y"] = tmp["y"].astype("float64")
        frames.append(tmp)

    combined = pd.concat(frames)
    aggregated = (
        combined.groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return aggregated


def fetch_rps(
    days_ago: int = 7,
    step: str = "5min",
    window: str = "5m",
    prometheus_url: str = "http://localhost:9090",
    token: str = "",
) -> pd.DataFrame:
    """
    Удобная обёртка: получает RPS с бэкенда grafana-deploy.

    Формула: sum(rate(http_requests_total[<window>]))
    - rate() вычисляет скорость изменения счётчика (запросов/сек)
    - sum() суммирует по всем path/method/status_code

    Parameters
    ----------
    window : окно для rate() в формате Prometheus ("5m", "1m", "1h")
             Должно быть ≥ step, обычно = 2*step или больше
    """
    query = f"sum(rate(litestar_requests_total[{window}]))"
    return fetch_metric(
        query=query,
        days_ago=days_ago,
        step=step,
        prometheus_url=prometheus_url,
        token=token,
    )


def fetch_latency_p99(
    days_ago: int = 7,
    step: str = "5min",
    prometheus_url: str = "http://localhost:9090",
    token: str = "",
) -> pd.DataFrame:
    """
    99-й перцентиль времени ответа (latency P99).
    Полезно как дополнительный признак в feature engineering.
    """
    query = (
        "histogram_quantile(0.99, "
        "sum(rate(litestar_request_duration_seconds_bucket[5m])) by (le))"
    )
    return fetch_metric(
        query=query,
        days_ago=days_ago,
        step=step,
        prometheus_url=prometheus_url,
        token=token,
    )


def fetch_active_connections(
    days_ago: int = 7,
    step: str = "5min",
    prometheus_url: str = "http://localhost:9090",
    token: str = "",
) -> pd.DataFrame:
    """
    Количество активных запросов (in-progress).
    """
    query = "sum(litestar_requests_in_progress)"
    return fetch_metric(
        query=query,
        days_ago=days_ago,
        step=step,
        prometheus_url=prometheus_url,
        token=token,
    )
