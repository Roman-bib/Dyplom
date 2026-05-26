"""
Streamlit-прототип системы прогнозирования пиковых нагрузок.
Запуск: streamlit run app.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

# ── Конфигурация страницы ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Прогнозирование нагрузки",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Импорт модулей проекта ───────────────────────────────────────────────────
try:
    import config
    from data_collection.csv_loader import load_csv, load_web_traffic
    from preprocessing.feature_engineering import FeatureBuilder, split_train_val_test
    from models.xgboost_model import train_xgboost, predict_xgboost, get_confidence_interval, feature_importance
    from evaluation.metrics import evaluate, safe_mape
    from evaluation.peak_detection import PeakDetector
    MODULES_OK = True
except Exception as e:
    MODULES_OK = False
    st.error(f"Ошибка импорта модулей проекта: {e}")
    st.stop()

SAVE_DIR = config.MODEL_SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

DAYS_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def dl(df: pd.DataFrame, filename: str, label: str = "⬇ Скачать CSV"):
    """Кнопка скачивания DataFrame как CSV."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3498db;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
    div[data-testid="stSidebarNav"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Настройки")

# ── Источник данных ──────────────────────────────────────────────────────────
st.sidebar.header("📂 Данные")
data_source = st.sidebar.radio(
    "Источник",
    ["web_traffic.csv (по умолчанию)", "Загрузить CSV"],
    index=0,
)

uploaded_file = None
if data_source == "Загрузить CSV":
    uploaded_file = st.sidebar.file_uploader("Выберите CSV", type=["csv"])

months = st.sidebar.slider("Месяцев данных", 1, 12, 12) if data_source == "web_traffic.csv (по умолчанию)" else None


@st.cache_data(show_spinner="Загрузка данных...")
def load_data(source: str, file_bytes=None, months: int = 12):
    if source == "upload" and file_bytes is not None:
        import io
        raw = pd.read_csv(io.BytesIO(file_bytes))
        return raw, list(raw.columns)
    else:
        df = load_web_traffic(months=months)
        # Восстанавливаем оригинальный CSV для отображения всех колонок
        try:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "Code", "data", "web_traffic.csv"
            )
            raw = pd.read_csv(default_path)
            raw = raw.iloc[:len(df) * 1]  # синхронизируем по длине
        except Exception:
            raw = df.rename(columns={"ds": "timestamp", "y": "rps"})
        return raw, list(raw.columns)

# Загружаем данные
if data_source == "Загрузить CSV" and uploaded_file is None:
    st.info("👆 Загрузите CSV-файл в боковой панели")
    st.stop()

file_bytes = uploaded_file.read() if uploaded_file else None
raw_df, all_columns = load_data(
    "upload" if uploaded_file else "default",
    file_bytes=file_bytes,
    months=months or 12,
)

# ── Выбор колонок ───────────────────────────────────────────────────────────
st.sidebar.header("📈 Метрика для прогноза")

# Определяем какие колонки являются datetime
ts_candidates = []
for col in all_columns:
    try:
        parsed = pd.to_datetime(raw_df[col])
        if parsed.notna().mean() > 0.9:
            ts_candidates.append(col)
    except Exception:
        pass

# Если datetime-колонок нет — предлагаем сгенерировать временную ось
has_datetime_col = len(ts_candidates) > 0

if not has_datetime_col:
    st.sidebar.info("⚠️ Колонка с датами не найдена — будет сгенерирована временная ось")

ts_mode = st.sidebar.radio(
    "Временная ось",
    ["Из колонки (datetime)", "Сгенерировать из индекса"],
    index=0 if has_datetime_col else 1,
)

ts_generate = False
ts_start = "2023-01-01"
ts_freq = "1h"

if ts_mode == "Из колонки (datetime)":
    ts_col = st.sidebar.selectbox(
        "Колонка времени",
        ts_candidates if ts_candidates else all_columns,
        index=0,
    )
else:
    ts_col = st.sidebar.selectbox(
        "Колонка-индекс (числовой порядок)",
        all_columns,
        index=0,
        help="Целые числа 0,1,2,3... или любой порядковый столбец",
    )
    ts_generate = True
    ts_start = st.sidebar.text_input("Стартовая дата", value="2023-01-01")
    ts_freq = st.sidebar.selectbox(
        "Частота",
        ["1h", "30min", "15min", "5min", "1min", "1D"],
        index=0,
        help="Шаг между точками",
    )

numeric_cols = [
    c for c in all_columns
    if c != ts_col and pd.api.types.is_numeric_dtype(raw_df[c])
]
default_val_idx = numeric_cols.index("rps") if "rps" in numeric_cols else 0

value_col = st.sidebar.selectbox(
    "Метрика (целевая переменная)",
    numeric_cols,
    index=default_val_idx,
    help="Эту метрику модель будет прогнозировать",
)

# ── Настройки моделей ────────────────────────────────────────────────────────
st.sidebar.header("🤖 Модели")
use_xgb     = st.sidebar.checkbox("XGBoost", value=True)
use_lstm    = st.sidebar.checkbox("LSTM", value=False)
use_prophet = st.sidebar.checkbox("NeuralProphet", value=False)
use_ci      = st.sidebar.checkbox("Доверительный интервал XGBoost", value=True)
use_linear  = False

# ── Кнопка обучения ──────────────────────────────────────────────────────────
st.sidebar.divider()
train_btn = st.sidebar.button("▶ Обучить модели", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ПОДГОТОВКА ДАННЫХ
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def prepare_df(file_bytes, source, ts_col, value_col, months,
               ts_generate=False, ts_start="2023-01-01", ts_freq="1h"):
    """Приводим к формату {ds, y}."""
    if source == "upload":
        import io
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw, _ = load_data("default", months=months)

    raw = raw.dropna(subset=[value_col]).reset_index(drop=True)

    if ts_generate:
        # Генерируем временную ось из стартовой даты и частоты
        raw["__ds__"] = pd.date_range(
            start=ts_start, periods=len(raw), freq=ts_freq
        )
        df = raw[["__ds__", value_col]].rename(
            columns={"__ds__": "ds", value_col: "y"}
        )
    else:
        raw[ts_col] = pd.to_datetime(raw[ts_col])
        df = raw[[ts_col, value_col]].rename(columns={ts_col: "ds", value_col: "y"})

    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = df["y"].astype(float)
    return df


df = prepare_df(
    file_bytes,
    "upload" if uploaded_file else "default",
    ts_col, value_col, months or 12,
    ts_generate=ts_generate,
    ts_start=ts_start,
    ts_freq=ts_freq,
)

n = len(df)
TEST_P = min(480, n // 6)
VAL_P  = TEST_P
train, val, test = split_train_val_test(df, test_hours=TEST_P, val_hours=VAL_P)


# ════════════════════════════════════════════════════════════════════════════
# ЗАГОЛОВОК
# ════════════════════════════════════════════════════════════════════════════

st.title("📈 Прогнозирование пиковых нагрузок")
st.caption(f"Метрика: **{value_col}** · Точек: **{n}** · "
           f"Период: {df['ds'].iloc[0].date()} – {df['ds'].iloc[-1].date()}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего точек", f"{n:,}")
col2.metric(f"Min {value_col}", f"{df['y'].min():.1f}")
col3.metric(f"Max {value_col}", f"{df['y'].max():.1f}")
col4.metric(f"Mean {value_col}", f"{df['y'].mean():.1f}")

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# ВКЛАДКИ
# ════════════════════════════════════════════════════════════════════════════

tab_eda, tab_split, tab_train, tab_forecast, tab_peaks, tab_importance, tab_retrain = st.tabs([
    "📊 EDA",
    "✂️ Разбиение",
    "🏆 Сравнение моделей",
    "📉 Прогноз",
    "🔔 Детекция пиков",
    "🔍 Признаки",
    "🔄 Адаптивное переобучение",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: EDA
# ════════════════════════════════════════════════════════════════════════════

with tab_eda:
    st.subheader(f"Временной ряд: {value_col}")

    # Скользящее среднее
    window = max(1, n // 100)
    df_plot = df.copy()
    df_plot["rolling_mean"] = df_plot["y"].rolling(window).mean()
    df_plot["rolling_std"]  = df_plot["y"].rolling(window).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["y"], name="Факт",
                             line=dict(color="#3498db", width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["rolling_mean"],
                             name=f"Скользящее среднее (окно={window})",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_plot["ds"], df_plot["ds"][::-1]]),
        y=pd.concat([
            df_plot["rolling_mean"] + df_plot["rolling_std"],
            (df_plot["rolling_mean"] - df_plot["rolling_std"])[::-1]
        ]),
        fill="toself", fillcolor="rgba(231,76,60,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="±1σ",
    ))
    fig.update_layout(height=380, xaxis_title="Время", yaxis_title=value_col,
                      legend=dict(orientation="h", y=1.1), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    dl(df_plot[["ds", "y", "rolling_mean", "rolling_std"]].round(4),
       "eda_timeseries.csv", "⬇ Временной ряд + скользящее среднее")

    # Распределение + перцентили
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Распределение")
        fig2 = px.histogram(df, x="y", nbins=60, color_discrete_sequence=["#3498db"])
        fig2.add_vline(x=df["y"].mean(), line_dash="dash", line_color="red",
                       annotation_text=f"Mean={df['y'].mean():.1f}")
        fig2.add_vline(x=df["y"].median(), line_dash="dot", line_color="orange",
                       annotation_text=f"Median={df['y'].median():.1f}")
        fig2.update_layout(height=300, xaxis_title=value_col, yaxis_title="Частота",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        dl(df[["y"]].rename(columns={"y": value_col}), "eda_distribution.csv",
           "⬇ Распределение")

    with c2:
        st.subheader("Перцентили")
        quantiles = [50, 75, 90, 95, 99]
        q_vals = [np.percentile(df["y"].dropna(), q) for q in quantiles]
        fig3 = px.bar(x=q_vals, y=[f"P{q}" for q in quantiles],
                      orientation="h", color_discrete_sequence=["#3498db"],
                      text=[f"{v:.1f}" for v in q_vals])
        fig3.update_layout(height=300, xaxis_title=value_col, yaxis_title="",
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        q_df = pd.DataFrame({"Перцентиль": [f"P{q}" for q in quantiles],
                              value_col: [round(v, 4) for v in q_vals]})
        dl(q_df, "eda_percentiles.csv", "⬇ Перцентили")

    # Сезонность
    st.subheader("Сезонность")
    df_s = df.copy()
    df_s["hour"]        = df_s["ds"].dt.hour
    df_s["day_of_week"] = df_s["ds"].dt.dayofweek

    c3, c4 = st.columns(2)
    with c3:
        hourly = df_s.groupby("hour")["y"].mean().reset_index()
        fig4 = px.line(hourly, x="hour", y="y", markers=True,
                       color_discrete_sequence=["#3498db"])
        fig4.update_layout(title="Суточный профиль (среднее)",
                           xaxis_title="Час суток", yaxis_title=value_col,
                           height=280)
        st.plotly_chart(fig4, use_container_width=True)
        dl(hourly.rename(columns={"hour": "Час", "y": value_col}).round(4),
           "eda_hourly_profile.csv", "⬇ Суточный профиль")

    with c4:
        daily = df_s.groupby("day_of_week")["y"].mean().reset_index()
        daily["day_name"] = daily["day_of_week"].map(lambda i: DAYS_LABELS[i])
        colors = ["#e74c3c" if i >= 5 else "#3498db" for i in daily["day_of_week"]]
        fig5 = px.bar(daily, x="day_name", y="y", color_discrete_sequence=["#3498db"])
        fig5.update_traces(marker_color=colors)
        fig5.update_layout(title="Средняя нагрузка по дням недели",
                           xaxis_title="", yaxis_title=value_col, height=280)
        st.plotly_chart(fig5, use_container_width=True)
        dl(daily[["day_name", "y"]].rename(columns={"day_name": "День", "y": value_col}).round(4),
           "eda_daily_profile.csv", "⬇ Дневной профиль")

    # Тепловая карта
    pivot = df_s.groupby(["day_of_week", "hour"])["y"].mean().unstack(fill_value=0)
    pivot.index = [DAYS_LABELS[i] for i in pivot.index]
    fig6 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                     labels=dict(x="Час", y="День", color=value_col),
                     aspect="auto")
    fig6.update_layout(title="Тепловая карта: день × час", height=300)
    st.plotly_chart(fig6, use_container_width=True)
    dl(pivot.reset_index().rename(columns={"index": "День"}),
       "eda_heatmap.csv", "⬇ Тепловая карта")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: РАЗБИЕНИЕ
# ════════════════════════════════════════════════════════════════════════════

with tab_split:
    st.subheader("Хронологическое разбиение (без перемешивания)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Train", f"{len(train):,} точек", f"{len(train)/n:.0%}")
    c2.metric("Validation", f"{len(val):,} точек", f"{len(val)/n:.0%}")
    c3.metric("Test", f"{len(test):,} точек", f"{len(test)/n:.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train",
                             line=dict(color="#3498db", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=val["ds"], y=val["y"], name="Validation",
                             line=dict(color="#f39c12", width=1.2)))
    fig.add_trace(go.Scatter(x=test["ds"], y=test["y"], name="Test",
                             line=dict(color="#27ae60", width=1.5)))
    fig.add_vrect(x0=val["ds"].iloc[0], x1=val["ds"].iloc[-1],
                  fillcolor="orange", opacity=0.07, line_width=0,
                  annotation_text="Val", annotation_position="top left")
    fig.add_vrect(x0=test["ds"].iloc[0], x1=test["ds"].iloc[-1],
                  fillcolor="green", opacity=0.08, line_width=0,
                  annotation_text="Test", annotation_position="top left")
    fig.update_layout(height=420, xaxis_title="Время", yaxis_title=value_col,
                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    split_df = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ]).rename(columns={"ds": "timestamp", "y": value_col})
    dl(split_df, "split_data.csv", "⬇ Данные с разбиением")


# ════════════════════════════════════════════════════════════════════════════
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════════════════

def run_training():
    builder = FeatureBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        builder.transform_splits(train, val, test)

    results   = {}
    models    = {}
    preds_all = {}
    lower_ci  = None
    upper_ci  = None

    progress = st.progress(0, text="Подготовка признаков...")
    step = 0
    total_steps = sum([use_xgb, use_lstm, use_prophet]) + (1 if use_ci and use_xgb else 0)

    if use_xgb:
        progress.progress(step / max(total_steps, 1), "XGBoost...")
        m = train_xgboost(X_train, y_train, X_val, y_val,
                          n_estimators=config.XGB_N_ESTIMATORS,
                          max_depth=config.XGB_MAX_DEPTH,
                          learning_rate=config.XGB_LEARNING_RATE,
                          early_stopping_rounds=config.XGB_EARLY_STOPPING,
                          save_path=os.path.join(SAVE_DIR, "xgboost.pkl"))
        p = predict_xgboost(m, X_test)
        results["XGBoost"] = evaluate(y_test.values, p, verbose=False)
        models["XGBoost"]  = m
        preds_all["XGBoost"] = p
        step += 1

        if use_ci:
            progress.progress(step / max(total_steps, 1), "Доверительный интервал...")
            lower_ci, upper_ci = get_confidence_interval(
                X_train, y_train, X_val, y_val, X_test,
                save_dir=SAVE_DIR,
            )
            step += 1

    if use_lstm:
        progress.progress(step / max(total_steps, 1), "LSTM...")
        try:
            from models.lstm_model import train_lstm, predict_lstm_aligned
            from preprocessing.feature_engineering import _infer_step_minutes, _hours_to_periods
            step_min = _infer_step_minutes(y_train.index)
            window_size = _hours_to_periods(24, step_min)
            window_size = max(12, min(window_size, 288))
            artifact = train_lstm(
                X_train, y_train, X_val, y_val,
                window_size=window_size,
                save_path=os.path.join(SAVE_DIR, "lstm"),
            )
            X_full = pd.concat([X_train, X_val, X_test])
            p = predict_lstm_aligned(artifact, X_full, n_test=len(X_test))
            results["LSTM"] = evaluate(y_test.values, p, verbose=False)
            models["LSTM"]  = artifact
            preds_all["LSTM"] = p
        except Exception as e:
            st.warning(f"LSTM: {e}")
        step += 1

    if use_prophet:
        progress.progress(step / max(total_steps, 1), "NeuralProphet...")
        try:
            from models.neural_prophet_model import train_neural_prophet, predict_neural_prophet
            train_val = pd.concat([train, val]).sort_values("ds")
            m = train_neural_prophet(
                train_df=train, val_df=val,
                save_path=os.path.join(SAVE_DIR, "neural_prophet"),
            )
            p = predict_neural_prophet(m, train_val_df=train_val, test_df=test)
            p = p[:len(y_test)]
            results["NeuralProphet"] = evaluate(y_test.values, p, verbose=False)
            models["NeuralProphet"] = m
            preds_all["NeuralProphet"] = p
        except Exception as e:
            st.warning(f"NeuralProphet: {e}")
        step += 1

    progress.progress(1.0, "Готово!")

    imp_df = None
    if "XGBoost" in models:
        imp_df = feature_importance(models["XGBoost"],
                                    feature_names=list(X_train.columns))

    return {
        "results": results,
        "models": models,
        "preds": preds_all,
        "y_test": y_test,
        "X_train": X_train,
        "lower": lower_ci,
        "upper": upper_ci,
        "imp_df": imp_df,
        "builder": builder,
    }


if train_btn:
    if not any([use_xgb, use_lstm, use_prophet]):
        st.warning("Выберите хотя бы одну модель")
    else:
        with st.spinner("Обучение моделей..."):
            st.session_state["trained"] = run_training()
        st.success("✅ Обучение завершено!")
        st.rerun()

trained = st.session_state.get("trained")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: СРАВНЕНИЕ МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════════════════

with tab_train:
    if trained is None:
        st.info("Нажмите **▶ Обучить модели** в боковой панели")
    else:
        results = trained["results"]
        best = min(results, key=lambda k: results[k]["MAPE"])

        st.subheader("Метрики качества")

        # Таблица
        rows = []
        for name, m in results.items():
            rows.append({
                "Модель": f"{'★ ' if name == best else ''}{name}",
                "MAE": round(m["MAE"], 3),
                "RMSE": round(m["RMSE"], 3),
                "MAPE, %": round(m["MAPE"], 2),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Модель"), use_container_width=True)

        # Bar charts
        model_names = list(results.keys())
        colors = ["#e74c3c" if n == best else "#3498db" for n in model_names]

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["MAE", "RMSE", "MAPE (%)"])
        for i, metric in enumerate(["MAE", "RMSE", "MAPE"], 1):
            vals = [results[m][metric] for m in model_names]
            fig.add_trace(go.Bar(x=model_names, y=vals, marker_color=colors,
                                 text=[f"{v:.2f}" for v in vals],
                                 textposition="outside", showlegend=False),
                          row=1, col=i)
        fig.update_layout(height=350, title_text="Сравнение метрик (меньше = лучше)")
        st.plotly_chart(fig, use_container_width=True)
        dl(pd.DataFrame(rows).set_index("Модель").reset_index(),
           "metrics_comparison.csv", "⬇ Метрики сравнения")

        st.success(f"🏆 Лучшая модель по MAPE: **{best}** ({results[best]['MAPE']:.2f}%)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: ПРОГНОЗ
# ════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    if trained is None:
        st.info("Нажмите **▶ Обучить модели** в боковой панели")
    else:
        preds_all = trained["preds"]
        y_test    = trained["y_test"]
        lower_ci  = trained["lower"]
        upper_ci  = trained["upper"]

        zoom = st.toggle("Только тест-период", value=True)
        show_ci = st.checkbox("Показать доверительный интервал (XGBoost)", value=True)

        model_colors = {
            "XGBoost": "#e74c3c",
            "LSTM": "#9b59b6",
            "NeuralProphet": "#e67e22",
        }

        fig = go.Figure()

        if not zoom:
            fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train",
                                     line=dict(color="#3498db", width=0.8),
                                     opacity=0.5))
            fig.add_trace(go.Scatter(x=val["ds"], y=val["y"], name="Validation",
                                     line=dict(color="#f39c12", width=0.8),
                                     opacity=0.7))

        n_test = min(len(test["ds"]), len(y_test))
        test_ds = test["ds"].values[:n_test]

        fig.add_trace(go.Scatter(x=test_ds, y=y_test.values[:n_test],
                                 name="Факт", line=dict(color="#27ae60", width=2),
                                 zorder=10))

        for name, preds in preds_all.items():
            n_p = min(n_test, len(preds))
            color = model_colors.get(name, "#333333")
            fig.add_trace(go.Scatter(x=test_ds[:n_p], y=preds[:n_p],
                                     name=f"{name} (прогноз)",
                                     line=dict(color=color, width=1.8, dash="dash")))

        if show_ci and lower_ci is not None and upper_ci is not None and "XGBoost" in preds_all:
            n_ci = min(n_test, len(lower_ci), len(upper_ci))
            coverage = np.mean(
                (y_test.values[:n_ci] >= lower_ci[:n_ci]) &
                (y_test.values[:n_ci] <= upper_ci[:n_ci])
            )
            fig.add_trace(go.Scatter(
                x=np.concatenate([test_ds[:n_ci], test_ds[:n_ci][::-1]]),
                y=np.concatenate([upper_ci[:n_ci], lower_ci[:n_ci][::-1]]),
                fill="toself", fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"ДИ 80% XGBoost (покрытие {coverage:.0%})",
            ))

        fig.update_layout(height=480, xaxis_title="Время", yaxis_title=value_col,
                          hovermode="x unified", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        # Экспорт прогнозов всех моделей
        forecast_df = pd.DataFrame({"timestamp": test_ds, "fact": y_test.values[:n_test]})
        for name, preds in preds_all.items():
            n_p = min(n_test, len(preds))
            forecast_df[name] = np.nan
            forecast_df.loc[:n_p-1, name] = preds[:n_p]
        if lower_ci is not None and upper_ci is not None:
            n_ci = min(n_test, len(lower_ci), len(upper_ci))
            forecast_df["XGBoost_lower"] = np.nan
            forecast_df["XGBoost_upper"] = np.nan
            forecast_df.loc[:n_ci-1, "XGBoost_lower"] = lower_ci[:n_ci]
            forecast_df.loc[:n_ci-1, "XGBoost_upper"] = upper_ci[:n_ci]
        dl(forecast_df.round(4), "forecast_predictions.csv", "⬇ Прогнозы всех моделей")

        # Таблица ошибок по периодам
        if "XGBoost" in preds_all:
            st.subheader("Абсолютная ошибка XGBoost (первые 48 точек)")
            n_show = min(48, n_test, len(preds_all["XGBoost"]))
            err_df = pd.DataFrame({
                "Время": test["ds"].values[:n_show],
                "Факт": y_test.values[:n_show].round(2),
                "Прогноз": preds_all["XGBoost"][:n_show].round(2),
                "Ошибка": np.abs(y_test.values[:n_show] - preds_all["XGBoost"][:n_show]).round(2),
                "Ошибка %": (np.abs(y_test.values[:n_show] - preds_all["XGBoost"][:n_show]) /
                              np.maximum(y_test.values[:n_show], 1e-6) * 100).round(1),
            })
            st.dataframe(err_df, use_container_width=True, height=300)
            dl(err_df, "forecast_errors.csv", "⬇ Таблица ошибок")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5: ДЕТЕКЦИЯ ПИКОВ
# ════════════════════════════════════════════════════════════════════════════

with tab_peaks:
    if trained is None:
        st.info("Нажмите **▶ Обучить модели** в боковой панели")
    elif "XGBoost" not in trained["preds"]:
        st.warning("Детекция пиков требует XGBoost — включите его в настройках")
    else:
        preds_xgb = trained["preds"]["XGBoost"]
        y_test    = trained["y_test"]

        c1, c2 = st.columns(2)
        peak_method = c1.selectbox("Метод", ["rolling_std", "percentile"])
        peak_k      = c2.slider("Коэффициент k (rolling_std)", 1.0, 4.0, 2.0, 0.1)

        rps_max = float(train["y"].max())
        target_per_replica = rps_max / config.MAX_REPLICAS

        detector = PeakDetector(
            method=peak_method, k=peak_k,
            target_rps_per_replica=target_per_replica,
            min_replicas=config.MIN_REPLICAS,
            max_replicas=config.MAX_REPLICAS,
            warning_ratio=config.ALERT_WARNING_RATIO,
            critical_ratio=config.ALERT_CRITICAL_RATIO,
        )
        detector.fit(train["y"])

        n_test = min(len(test["ds"]), len(y_test), len(preds_xgb))
        predicted_series = pd.Series(preds_xgb[:n_test], index=y_test.index[:n_test])
        events_df = detector.detect_series(y_test.iloc[:n_test], predicted_series)
        summary = detector.summary(events_df)

        # Карточки
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Порог", f"{summary['threshold']:.1f}")
        c2.metric("Пиков", f"{summary['peaks_detected']}")
        c3.metric("Доля пиков", f"{summary['peak_ratio_pct']}%")
        c4.metric("Critical", summary["severity_counts"].get("critical", 0))

        # График
        sev_color = {"ok": "#27ae60", "info": "#3498db",
                     "warning": "#f39c12", "critical": "#e74c3c"}

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=["Прогноз vs факт + пороги", "Severity"])

        fig.add_trace(go.Scatter(x=events_df["timestamp"], y=events_df["rps"],
                                 name="Факт", line=dict(color="#27ae60", width=1.8)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=events_df["timestamp"], y=events_df["predicted"],
                                 name="Прогноз XGBoost",
                                 line=dict(color="#e74c3c", width=1.5, dash="dash")),
                      row=1, col=1)
        fig.add_hline(y=detector.threshold, line_dash="dot", line_color="black",
                      annotation_text=f"Порог ({detector.threshold:.0f})", row=1, col=1)
        fig.add_hline(y=detector.threshold * config.ALERT_WARNING_RATIO,
                      line_dash="dot", line_color="#f39c12",
                      annotation_text="Warning", row=1, col=1)

        sev_map = {"ok": 0, "info": 1, "warning": 2, "critical": 3}
        fig.add_trace(go.Scatter(
            x=events_df["timestamp"],
            y=events_df["severity"].map(sev_map),
            mode="markers",
            marker=dict(
                color=[sev_color[s] for s in events_df["severity"]],
                size=4,
            ),
            name="Severity",
        ), row=2, col=1)
        fig.update_yaxes(tickvals=[0, 1, 2, 3],
                         ticktext=["ok", "info", "warning", "critical"],
                         row=2, col=1)
        fig.update_layout(height=550, hovermode="x unified",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        dl(events_df.round(4), "peak_events_all.csv", "⬇ Все события (факт + прогноз + severity)")

        # Таблица пиков
        peaks = events_df[events_df["is_peak"]].copy()
        if len(peaks) > 0:
            st.subheader(f"Обнаруженные пики ({len(peaks)} шт.)")
            peaks_show = peaks[["timestamp", "rps", "predicted", "severity",
                                "recommended_replicas"]].copy()
            peaks_show.columns = ["Время", "Факт", "Прогноз", "Severity", "Реплик"]
            peaks_show["Факт"]    = peaks_show["Факт"].round(1)
            peaks_show["Прогноз"] = peaks_show["Прогноз"].round(1)
            st.dataframe(peaks_show.reset_index(drop=True),
                         use_container_width=True, height=300)
            dl(peaks_show.reset_index(drop=True), "peak_events_only.csv", "⬇ Только пики")

        # Рекомендации по репликам
        st.subheader("Рекомендуемое число реплик (по прогнозу)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=events_df["timestamp"], y=events_df["recommended_replicas"],
            mode="lines", line=dict(color="#3498db", width=2, shape="hv"),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.15)", name="Реплик",
        ))
        fig2.add_hline(y=config.MIN_REPLICAS, line_dash="dash", line_color="green",
                       annotation_text=f"Min={config.MIN_REPLICAS}")
        fig2.add_hline(y=config.MAX_REPLICAS, line_dash="dash", line_color="red",
                       annotation_text=f"Max={config.MAX_REPLICAS}")
        fig2.update_layout(height=280, yaxis_title="Реплик",
                           yaxis=dict(dtick=1, range=[0, config.MAX_REPLICAS + 1]))
        st.plotly_chart(fig2, use_container_width=True)
        dl(events_df[["timestamp", "predicted", "recommended_replicas"]].round(4),
           "scaling_recommendations.csv", "⬇ Рекомендации по репликам")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6: ВАЖНОСТЬ ПРИЗНАКОВ
# ════════════════════════════════════════════════════════════════════════════

with tab_importance:
    if trained is None:
        st.info("Нажмите **▶ Обучить модели** в боковой панели")
    elif trained.get("imp_df") is None:
        st.warning("Включите XGBoost для получения важности признаков")
    else:
        imp_df = trained["imp_df"]
        st.subheader("Важность признаков XGBoost (gain)")

        norm = imp_df["importance"] / imp_df["importance"].sum() * 100
        imp_plot = imp_df.copy()
        imp_plot["importance_pct"] = norm.round(2)

        fig = px.bar(
            imp_plot.sort_values("importance"),
            x="importance_pct", y="feature",
            orientation="h",
            color="importance_pct",
            color_continuous_scale="RdYlGn",
            text="importance_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=420, xaxis_title="Важность (%)",
                          coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        dl(imp_plot[["feature", "importance_pct"]].rename(
               columns={"feature": "Признак", "importance_pct": "Важность, %"}),
           "feature_importance.csv", "⬇ Важность признаков")

        st.subheader("Корреляция признаков с целевой переменной")
        builder = trained["builder"]
        sample = builder.transform(df.tail(500)).dropna()
        avail = [c for c in builder.FEATURE_COLS if c in sample.columns]
        corr = sample[avail + ["y"]].corr()["y"].drop("y").sort_values()

        fig2 = px.bar(
            x=corr.values, y=corr.index,
            orientation="h",
            color=corr.values,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
        )
        fig2.update_layout(height=380, xaxis_title="Корреляция Пирсона",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        corr_df = corr.reset_index()
        corr_df.columns = ["Признак", "Корреляция"]
        dl(corr_df.round(4), "feature_correlations.csv", "⬇ Корреляции признаков")

        st.subheader("Таблица признаков")
        imp_table = imp_plot[["feature", "importance_pct"]].rename(
            columns={"feature": "Признак", "importance_pct": "Важность, %"}
        ).reset_index(drop=True)
        st.dataframe(imp_table, use_container_width=True)
        dl(imp_table, "feature_importance_full.csv", "⬇ Полная таблица признаков")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7: АДАПТИВНОЕ ПЕРЕОБУЧЕНИЕ (walk-forward симуляция)
# ════════════════════════════════════════════════════════════════════════════

with tab_retrain:
    st.subheader("Адаптивное переобучение: walk-forward симуляция")
    st.caption(
        "Система прогнозирует точку за точкой (walk-forward). При обнаружении "
        "концепт-дрейфа через **ADWIN** или по счётчику **N_fresh** — модель "
        "автоматически переобучается на накопленной истории. Параллельно идёт "
        "фиксированная baseline-модель (без переобучений). "
        "Это точная реализация алгоритма из блок-схемы (Рисунок 2.12 ВКР)."
    )

    c1, c2, c3 = st.columns(3)
    n_fresh_val = c1.slider(
        "N_fresh: переобучение каждые N шагов",
        min_value=0, max_value=500, value=200, step=10,
        help="0 = только по ADWIN-сигналу (без принудительного)",
    )
    confirm_n_val = c2.slider(
        "Подтверждение ADWIN (шагов подряд)",
        min_value=3, max_value=30, value=10,
        help="Защита от разовых пиков: retrain только если drift держится N шагов",
    )
    test_limit = c3.slider(
        "Точек в тесте",
        min_value=100, max_value=min(1000, len(test)),
        value=min(400, len(test)), step=50,
        help="Больше точек = дольше работает (~1-2 с на 100 точек)",
    )

    if trained is None or "XGBoost" not in trained.get("models", {}):
        st.warning("⚠️ Сначала обучите **XGBoost** на вкладке «Сравнение моделей»")
    elif st.button("▶ Запустить walk-forward", type="primary"):
        with st.spinner("Walk-forward симуляция..."):
            try:
                from evaluation.walk_forward import run_walk_forward
                from retraining.scheduler import make_xgb_train_fn
                from retraining.drift_detector import ADWINDriftDetector
                from models.forecasters import predict_xgboost_wf

                builder_wf = FeatureBuilder()
                drift_det  = ADWINDriftDetector(
                    n_fresh=n_fresh_val,
                    confirmation_n=confirm_n_val,
                )
                train_fn_wf = make_xgb_train_fn(builder_wf, save_dir=SAVE_DIR)
                test_wf = test.iloc[:test_limit].reset_index(drop=True)

                wf_res = run_walk_forward(
                    train=train,
                    val=val,
                    test=test_wf,
                    initial_model=trained["models"]["XGBoost"],
                    predict_fn=predict_xgboost_wf,
                    train_fn=train_fn_wf,
                    builder=builder_wf,
                    drift_detector=drift_det,
                    save_dir=SAVE_DIR,
                    verbose=False,
                )
                st.session_state["wf_res"] = wf_res
                st.success("✅ Готово!")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    wf = st.session_state.get("wf_res")

    if wf is not None:
        results_df  = wf["results_df"]
        baseline_df = wf["baseline_df"]
        summary     = wf["summary"]
        retrain_ts  = summary["retrain_timestamps"]

        # --- Сводные метрики ---
        st.subheader("Итоговые метрики")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE адаптивная",    f"{summary['mae_adaptive']:.3f}")
        c2.metric("MAE фиксированная", f"{summary['mae_baseline']:.3f}",
                  delta=f"{summary['mae_adaptive'] - summary['mae_baseline']:+.3f}",
                  delta_color="inverse")
        c3.metric("Улучшение",         f"{summary['improvement_pct']:.1f}%")
        c4.metric("Переобучений",       str(summary['n_retrains']))

        # --- График 1: прогноз vs факт ---
        st.subheader("Прогноз vs факт (walk-forward на тесте)")
        merged = results_df.merge(baseline_df, on="timestamp", how="left")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=merged["timestamp"], y=merged["y_true"],
            name="Факт", line=dict(color="#27ae60", width=1.5),
        ))
        fig1.add_trace(go.Scatter(
            x=merged["timestamp"], y=merged["y_pred"],
            name=f"Адаптивная (MAE={summary['mae_adaptive']:.2f})",
            line=dict(color="#3498db", width=1.5, dash="dash"),
        ))
        if "y_pred_baseline" in merged.columns:
            fig1.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["y_pred_baseline"],
                name=f"Фиксированная (MAE={summary['mae_baseline']:.2f})",
                line=dict(color="#e74c3c", width=1.2, dash="dot"),
            ))
        for i, ts in enumerate(retrain_ts):
            fig1.add_vline(
                x=ts, line_dash="dot", line_color="crimson",
                line_width=1.2, opacity=0.65,
                annotation_text="↺ retrain" if i == 0 else "↺",
                annotation_position="top",
            )
        fig1.update_layout(
            height=420, hovermode="x unified",
            xaxis_title="Время", yaxis_title=value_col,
            legend=dict(orientation="h", y=1.12),
            title="Walk-forward: адаптивный vs фиксированный прогноз  (↺ = переобучение)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- График 2: скользящая MAE ---
        st.subheader("Скользящая MAE: адаптивная vs фиксированная")
        roll_w = max(12, len(results_df) // 20)
        roll_adapt = results_df["mae"].rolling(roll_w, min_periods=1).mean()
        roll_base  = baseline_df["mae_baseline"].rolling(roll_w, min_periods=1).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=results_df["timestamp"], y=roll_adapt,
            name="Адаптивная", line=dict(color="#3498db", width=2),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
        ))
        fig2.add_trace(go.Scatter(
            x=baseline_df["timestamp"], y=roll_base,
            name="Фиксированная", line=dict(color="#e74c3c", width=2, dash="dash"),
        ))
        for i, ts in enumerate(retrain_ts):
            fig2.add_vline(
                x=ts, line_dash="dot", line_color="crimson",
                line_width=1.2, opacity=0.7,
                annotation_text="retrain" if i == 0 else "",
                annotation_position="top left",
            )
        fig2.update_layout(
            height=320, hovermode="x unified",
            xaxis_title="Время", yaxis_title=f"MAE (скользящее окно={roll_w})",
            legend=dict(orientation="h", y=1.12),
            title="Скользящая MAE: адаптивная vs фиксированная модель",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Таблица событий переобучения из audit-лога ---
        if retrain_ts:
            st.subheader(f"Лог переобучений ({len(retrain_ts)} событий)")
            audit_path = os.path.join(SAVE_DIR, "walk_forward_log.csv")
            if os.path.exists(audit_path):
                audit_df = pd.read_csv(audit_path).tail(len(retrain_ts) + 5)
                rename = {
                    "timestamp": "Время",
                    "reason": "Причина",
                    "baseline_mae_before": "MAE до",
                    "rolling_mae_before": "MAE скольз.",
                    "new_baseline_mae": "MAE после",
                    "train_size": "Размер истории",
                    "duration_s": "Время (с)",
                }
                audit_df = audit_df.rename(columns=rename)
                st.dataframe(audit_df.reset_index(drop=True), use_container_width=True)
                dl(audit_df, "retrain_log.csv", "⬇ Лог переобучений")

        # --- Экспорт результатов ---
        export_wf = merged[
            ["timestamp", "y_true", "y_pred", "mae", "mae_baseline", "retrain"]
        ].round(3)
        dl(export_wf, "walk_forward_results.csv", "⬇ Результаты walk-forward")
