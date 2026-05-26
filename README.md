# Proactive Scaler – Дипломный проект

Прогнозирование пиковых нагрузок и автоматическое масштабирование веб-сервера с использованием машинного обучения.

## 1. Подготовка окружения
```bash
git clone https://github.com/artemonsh/grafana-deploy.git
cd grafana-deploy
docker-compose up -d   # запускает бекенд, Prometheus, Grafana
cd ..
git clone <ваш-репозиторий>
cd proactive-scaler
python -m venv venv
source venv/bin/activate  # для Linux / Mac
venv\Scripts\activate     # для Windows
pip install -r requirements.txt
```

# Обучение моделей

```bash
python main.py train
```

Будут обучены XGBoost, Prophet, LSTM, ARIMA и сохранены в папку saved_models/.

# Симуляция проактивного масштабирования
Запустите в отдельном терминале нагрузочный тест Locust:
```bash
locust -f simulation/locustfile.py --host=http://localhost:8080
```
Затем запустите скейлер:

```bash
python main.py simulate
Он будет раз в минуту запрашивать метрики, прогнозировать нагрузку и выводить решения о масштабировании.
```

# Визуализация и метрики
Используйте дашборд Grafana из grafana-deploy (http://localhost:3000, логин/пароль admin/admin) или встроенные графики после обучения.

## ✅ Что дальше

1. Скопируйте все файлы в описанную выше структуру.
2. Установите Docker, запустите стенд мониторинга (`grafana-deploy`).
3. Запустите обучение моделей (`python main.py train`).
4. Запустите симуляцию и убедитесь, что скейлер реагирует на прогнозируемые пики.

Если что-то пойдет не так – проверьте доступность Prometheus (`http://localhost:9090`), метрику http_requests_total (в стенде она точно есть) и совместимость библиотек. Я готов помочь с отладкой.