"""
Locust-сценарий для генерации реалистичного трафика с пиками.

Бэкенд: github.com/artemonsh/grafana-deploy (Litestar на порту 8080)

Запуск без веб-интерфейса (headless), 10 пользователей, 7 дней данных:
    locust -f simulation/locustfile.py \\
           --host=http://localhost:8080 \\
           --headless -u 10 -r 2 \\
           --run-time 7d

Запуск с веб-интерфейсом (рекомендуется для демо):
    locust -f simulation/locustfile.py --host=http://localhost:8080
    # затем открыть http://localhost:8089 и управлять нагрузкой вручную

Профили нагрузки:
  NormalUser    — фоновый трафик (1-3 req/s на пользователя)
  SpikeUser     — пиковая нагрузка (запросы без паузы, ~10x выше нормы)
  ErrorUser     — ошибочные запросы (4xx/5xx) для проверки метрик
"""

import random

from locust import HttpUser, LoadTestShape, between, events, task


# ---------------------------------------------------------------------------
# Пользовательские классы
# ---------------------------------------------------------------------------

class NormalUser(HttpUser):
    """Обычный пользователь: равномерная фоновая нагрузка."""

    wait_time = between(0.5, 2.0)
    weight = 70                     # 70% пользователей — нормальные

    @task(5)
    def root(self):
        self.client.get("/", name="GET /")

    @task(3)
    def status_200(self):
        self.client.get("/status/200", name="GET /status/200")

    @task(2)
    def status_200_slow(self):
        # Медленный запрос — имитирует базу данных / внешний сервис
        delay = random.choice([1, 2])
        self.client.get(
            f"/status/200?seconds_sleep={delay}",
            name="GET /status/200 (slow)",
        )


class ErrorUser(HttpUser):
    """Пользователь, генерирующий ошибки (4xx/5xx)."""

    wait_time = between(1.0, 4.0)
    weight = 15                     # 15% пользователей

    @task(3)
    def error_404(self):
        self.client.get("/status/404", name="GET /status/4xx")

    @task(1)
    def error_500(self):
        self.client.get("/status/500", name="GET /status/5xx")


class SpikeUser(HttpUser):
    """Агрессивный пользователь — моделирует пиковый всплеск."""

    wait_time = between(0.05, 0.2)  # почти без паузы
    weight = 15                     # 15% пользователей

    @task(8)
    def burst_root(self):
        self.client.get("/", name="GET / [spike]")

    @task(2)
    def burst_status(self):
        self.client.get("/status/200", name="GET /status/200 [spike]")


# ---------------------------------------------------------------------------
# Форма нагрузки — имитирует паттерны трафика с пиками
# ---------------------------------------------------------------------------

class RealisticLoadShape(LoadTestShape):
    """
    Ступенчатый профиль нагрузки с периодическими пиками.

    Структура (по умолчанию — 45 минут):
      0–5 мин   : разогрев (2 пользователя)
      5–15 мин  : нормальная нагрузка (8 пользователей)
      15–20 мин : ПЕРВЫЙ ПИК (40 пользователей)
      20–25 мин : спад до нормы (8 пользователей)
      25–35 мин : нормальная нагрузка (8 пользователей)
      35–40 мин : ВТОРОЙ ПИК (60 пользователей)
      40–45 мин : спад (5 пользователей)
      45 мин    : стоп

    Это хорошо воспроизводит реальный суточный паттерн в ускоренном режиме
    и позволяет системе прогнозирования увидеть пики.
    """

    # (конец_секунды, user_count, spawn_rate)
    STAGES = [
        (300,   2,  1),    # 0–5 мин:    разогрев
        (900,   8,  2),    # 5–15 мин:   нормальный трафик
        (1200,  40, 10),   # 15–20 мин:  ПИК 1 (x5 нормы)
        (1500,  8,  5),    # 20–25 мин:  спад
        (2100,  8,  1),    # 25–35 мин:  нормальный трафик
        (2400,  60, 15),   # 35–40 мин:  ПИК 2 (x7.5 нормы)
        (2700,  5,  5),    # 40–45 мин:  финальный спад
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage_end, user_count, spawn_rate in self.STAGES:
            if run_time < stage_end:
                return user_count, spawn_rate

        return None  # стоп после всех стадий


# ---------------------------------------------------------------------------
# Событие: вывод информации о пиках в консоль
# ---------------------------------------------------------------------------

@events.test_start.add_listener
def on_test_start(**_):
    print("\n" + "=" * 60)
    print("Locust: генерация трафика с пиками запущена")
    print("Профиль: RealisticLoadShape (45 минут)")
    print("Пики: ~15-20 мин и ~35-40 мин от старта")
    print("Для мониторинга: http://localhost:3000 (Grafana)")
    print("Метрики: http://localhost:9090 (Prometheus)")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(**_):
    print("\nLocust: тест завершён. Данные доступны в Prometheus.\n")
