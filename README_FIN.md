# Полное руководство по proactive-scaler

---

## Предварительные условия

- Docker Desktop запущен
- Python 3.11 установлен
- venv создан и активирован (см. Шаг 1)

---

## Каждый раз при начале работы

### Активировать venv

```powershell
cd "C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler"

# Если venv ещё не создан:
py -3.11 -m venv venv311

# Активация:
.\venv311\Scripts\Activate.ps1
```

# Прогнозирование на реальных данных

## Шаг 1 — Запустить стек мониторинга

```powershell
cd grafana-deploy
docker-compose up -d
cd ..   # вернуться в proactive-scaler/
```

## Шаг 2 — Проверить, что всё поднялось

```powershell
docker ps
```

Должны быть **Up**: `backend`, `prometheus`, `grafana`, `loki`, `promtail`

---

## Генерация трафика (Locust)

```powershell
locust -f simulation/locustfile.py --host=http://localhost:8080
```

Открыть [http://localhost:8089](http://localhost:8089) → нажать **START**.

Тест идёт **45 минут** автоматически (пики на 15-й и 35-й минуте).

---

## ML-пайплайн (реальные данные из Prometheus)

Выполнять после того как накопилось **≥ 7 дней** данных в Prometheus (или сразу после первого теста для проверки).

```powershell
# Обучить модель на реальных данных:
python run_integration.py train

# Разовый прогноз на ближайший период:
python run_integration.py forecast

# Непрерывный мониторинг + авторекомендации по репликам:
python run_integration.py monitor

# Мониторинг с нестандартным интервалом опроса (секунды):
python run_integration.py monitor --interval 30

# Взять больше истории (по умолчанию 7 дней):
python run_integration.py train --days 14

# Пропустить Prophet (быстрее):
python run_integration.py train --no-prophet
```

---

# Обучение на синтетических данных

Принимает **любой CSV с временным рядом**. Требования:
- колонка с датой/временем (по умолчанию `timestamp`)
- колонка с числовым значением (по умолчанию `rps`)
- минимум **200 строк** (рекомендуется ≥ 7 дней данных)

### Формат CSV

```
timestamp,rps
2023-01-01 00:00:00,450
2023-01-01 01:00:00,512
2023-01-01 02:00:00,480
```

Остальные колонки (cpu_usage, latency_ms и т.д.) игнорируются.

### Команды

```powershell
# Стандартный файл web_traffic.csv (лежит в ../Code/data/):
python main.py csv

# Свой CSV — колонки определяются автоматически:
python main.py csv --path "C:\путь\к\файлу.csv"

# Явно указать колонки (если автоопределение ошиблось):
python main.py csv --path "C:\путь\к\файлу.csv" --timestamp-col "id_time" --value-col "n_flows"

# Только первые N месяцев данных:
python main.py csv --path "C:\путь\к\файлу.csv" --months 3

# Без Prophet (быстрее):
python main.py csv --path "C:\путь\к\файлу.csv" --fast

# Сохранить графики в saved_models/:
python main.py csv --path "C:\путь\к\файлу.csv" --save-plots
```

> Система автоматически находит datetime-колонку и первую числовую. При запуске выводит что выбрано: `timestamp → 'id_time',  value → 'n_flows'` — можно проверить и переопределить через флаги.

### Ограничения по данным

| Параметр | Минимум | Рекомендуется |
|---|---|---|
| Строк в файле | 200 | ≥ 8760 (год, шаг 1ч) |
| Период данных | 2 дня | ≥ 7 дней |
| Шаг данных | любой | 1h или 5min |

> Если данных меньше 7 дней — лаговый признак lag_168h (недельная сезонность) будет отсутствовать, модель обучится, но точность снизится.

---

## Демо-режим

```powershell
# Полное демо: обучение + сравнение всех моделей + графики:
python main.py demo

# Без Prophet (ускоряет в 3–5 раз):
python main.py demo --fast

# Только обучить XGBoost на синтетических данных:
python main.py train --synthetic

# Сравнить все модели (XGBoost, Ridge, Prophet, LSTM):
python main.py compare --synthetic

# Симуляция проактивного масштабирования:
python main.py simulate --synthetic
```

---

## Результаты обучения

После любого режима обучения файлы сохраняются в `saved_models/`:

| Файл | Содержимое |
|---|---|
| `xgboost.pkl` | Обученная модель XGBoost |
| `metrics_comparison.csv` | MAE / MAPE всех моделей |
| `predictions.csv` | Прогнозы на тестовом периоде |
| `feature_importance.csv` | Важность признаков XGBoost |
| `forecast_xgboost_full.png` | График прогноза (весь период) |
| `forecast_xgboost_zoom.png` | График прогноза (тестовый период, zoom) |

---

## Ссылки для мониторинга

| Сервис | URL | Логин |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Backend метрики | http://localhost:8080/metrics | — |
| Locust | http://localhost:8089 | — |

---

## Остановка стека

```powershell
cd grafana-deploy
docker-compose down
```

Данные Prometheus и Grafana сохраняются в Docker volumes — при следующем `up -d` восстановятся.
