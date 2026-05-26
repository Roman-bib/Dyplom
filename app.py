"""
Streamlit-прототип системы прогнозирования пиковых нагрузок.
Запуск: streamlit run app.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Прогнозирование нагрузки",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import config
    from data_collection.csv_loader import load_web_traffic
    from preprocessing.feature_engineering import FeatureBuilder, split_train_val_test
    from preprocessing.data_cleaning import TimeSeriesCleaner
    from models.forecasters import get_confidence_interval, feature_importance
    from models.comparison import ModelComparison
    from evaluation.peak_detection import PeakDetector
except Exception as e:
    st.error(f"Ошибка импорта модулей проекта: {e}")
    st.stop()

SAVE_DIR = config.MODEL_SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
DAYS_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def dl(df: pd.DataFrame, filename: str, label: str = "⬇ Скачать CSV"):
    st.download_button(label=label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", use_container_width=False)


st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Настройки")
st.sidebar.header("📂 Данные")
data_source = st.sidebar.radio("Источник", ["web_traffic.csv (по умолчанию)", "Загрузить CSV"], index=0)
uploaded_file = None
if data_source == "Загрузить CSV":
    uploaded_file = st.sidebar.file_uploader("Выберите CSV", type=["csv"])
months = st.sidebar.slider("Месяцев данных", 1, 12, 12) if data_source == "web_traffic.csv (по умолчанию)" else None


@st.cache_data(show_spinner="Загрузка данных...")
def load_data(source: str, file_bytes=None, months: int = 12):
    if source == "upload" and file_bytes is not None:
        import io
        buf = io.BytesIO(file_bytes)
        try:
            raw = pd.read_csv(buf, engine="python", encoding_errors="replace")
        except Exception:
            buf.seek(0)
            raw = pd.concat(list(pd.read_csv(buf, engine="python", encoding_errors="replace", chunksize=50_000)),
                            ignore_index=True)
        for col in raw.columns:
            if raw[col].dtype == object:
                raw[col] = raw[col].astype(str)
        return raw, list(raw.columns)
    else:
        df = load_web_traffic(months=months)
        try:
            default_path = os.path.join(os.path.dirname(__file__), "..", "Code", "data", "web_traffic.csv")
            raw = pd.read_csv(default_path).iloc[:len(df)]
        except Exception:
            raw = df.rename(columns={"ds": "timestamp", "y": "rps"})
        return raw, list(raw.columns)


if data_source == "Загрузить CSV" and uploaded_file is None:
    st.info("👆 Загрузите CSV-файл в боковой панели")
    st.stop()

file_bytes = uploaded_file.read() if uploaded_file else None
raw_df, all_columns = load_data(
    "upload" if uploaded_file else "default",
    file_bytes=file_bytes,
    months=months or 12,
)

# ── Выбор колонок ────────────────────────────────────────────────────────────
st.sidebar.header("📈 Метрика для прогноза")
ts_candidates = []
for col in all_columns:
    if pd.api.types.is_datetime64_any_dtype(raw_df[col]):
        ts_candidates.append(col); continue
    if pd.api.types.is_numeric_dtype(raw_df[col]):
        vals = raw_df[col].dropna()
        if len(vals) > 0:
            m = float(vals.median())
            if (1e9 <= m <= 2e9) or (1e12 <= m <= 2e15):
                ts_candidates.append(col)
        continue
    try:
        if pd.to_datetime(raw_df[col], errors="coerce").notna().mean() > 0.9:
            ts_candidates.append(col)
    except Exception:
        pass

has_datetime_col = bool(ts_candidates)
if not has_datetime_col:
    st.sidebar.info("⚠️ Колонка с датами не найдена — будет сгенерирована временная ось")

ts_mode = st.sidebar.radio("Временная ось",
                            ["Из колонки (datetime)", "Сгенерировать из индекса"],
                            index=0 if has_datetime_col else 1)
ts_generate, ts_start, ts_freq = False, "2023-01-01", "1h"
if ts_mode == "Из колонки (datetime)":
    ts_col = st.sidebar.selectbox("Колонка времени", ts_candidates or all_columns, index=0)
else:
    ts_col = st.sidebar.selectbox("Колонка-индекс", all_columns, index=0)
    ts_generate = True
    ts_start = st.sidebar.text_input("Стартовая дата", value="2023-01-01")
    ts_freq = st.sidebar.selectbox("Частота", ["1h", "30min", "15min", "5min", "1min", "1D"], index=0)

_all_numeric = [c for c in all_columns if c != ts_col and pd.api.types.is_numeric_dtype(raw_df[c])]
_preferred = [c for c in ["rps", "concurrent_users"] if c in _all_numeric]
numeric_cols = _preferred or _all_numeric
value_col = st.sidebar.selectbox("Метрика (целевая переменная)", numeric_cols,
                                  index=numeric_cols.index("rps") if "rps" in numeric_cols else 0)

# ── Разбиение + модели ────────────────────────────────────────────────────────
st.sidebar.header("📐 Разбиение данных")
test_pct = st.sidebar.slider("Test, %", 5, 30, 15, 1)
val_pct  = st.sidebar.slider("Validation, %", 5, 30, 15, 1)
st.sidebar.caption(f"Train будет: **{100 - test_pct - val_pct}%**")

st.sidebar.header("🤖 Модели")
use_xgb     = st.sidebar.checkbox("XGBoost", value=True)
use_lstm    = st.sidebar.checkbox("LSTM (нейросеть)", value=False,
                                   help="Требует TensorFlow и больше времени")
use_prophet = st.sidebar.checkbox("Prophet", value=False)
use_ci      = st.sidebar.checkbox("Доверительный интервал XGBoost", value=True)
use_upper_for_scaling = st.sidebar.checkbox(
    "Масштабировать по верхнему квантилю (P90)", value=True,
    help="Реагировать на ВЕРХНЮЮ границу прогноза (риск пика). Требует ДИ XGBoost.",
)
st.sidebar.divider()
train_btn = st.sidebar.button("▶ Обучить модели", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ПРЕДОБРАБОТКА
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _prepare_raw(file_bytes, source, ts_col, value_col, months,
                 ts_generate=False, ts_start="2023-01-01", ts_freq="1h"):
    if source == "upload":
        import io
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw, _ = load_data("default", months=months)
    raw = raw.dropna(subset=[value_col]).reset_index(drop=True)
    if ts_generate:
        raw["__ds__"] = pd.date_range(start=ts_start, periods=len(raw), freq=ts_freq)
        return raw[["__ds__", value_col]].rename(columns={"__ds__": "ds", value_col: "y"})
    return raw[[ts_col, value_col]].rename(columns={ts_col: "ds", value_col: "y"})


def prepare_df(file_bytes, source, ts_col, value_col, months,
               ts_generate=False, ts_start="2023-01-01", ts_freq="1h"):
    df_raw = _prepare_raw(file_bytes, source, ts_col, value_col, months,
                          ts_generate, ts_start, ts_freq)
    n = len(df_raw)
    p = max(60, int(n * 15 / 100))
    if p * 2 >= n - 100:
        p = max(60, (n - 100) // 4)
    cleaner = TimeSeriesCleaner()
    cleaner.fit(df_raw.iloc[:n - p * 2].copy())
    df_clean, _ = cleaner.transform(df_raw)
    return df_clean, cleaner


df, _cleaner = prepare_df(
    file_bytes, "upload" if uploaded_file else "default",
    ts_col, value_col, months or 12,
    ts_generate=ts_generate, ts_start=ts_start, ts_freq=ts_freq,
)

n = len(df)
TEST_P = max(60, int(n * test_pct / 100))
VAL_P  = max(60, int(n * val_pct  / 100))
if TEST_P + VAL_P >= n - 100:
    TEST_P = VAL_P = max(60, (n - 100) // 4)
train, val, test = split_train_val_test(df, test_hours=TEST_P, val_hours=VAL_P)

# ════════════════════════════════════════════════════════════════════════════
# ЗАГОЛОВОК
# ════════════════════════════════════════════════════════════════════════════

st.title("📈 Прогнозирование пиковых нагрузок")
st.caption(f"Метрика: **{value_col}** · Точек: **{n}** · "
           f"Период: {df['ds'].iloc[0].date()} – {df['ds'].iloc[-1].date()}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего точек", f"{n:,}")
c2.metric(f"Min {value_col}", f"{df['y'].min():.1f}")
c3.metric(f"Max {value_col}", f"{df['y'].max():.1f}")
c4.metric(f"Mean {value_col}", f"{df['y'].mean():.1f}")
st.divider()

tab_eda, tab_split, tab_train, tab_forecast, tab_peaks, tab_importance = st.tabs([
    "📊 EDA", "✂️ Разбиение", "🏆 Сравнение моделей",
    "📉 Прогноз", "🔔 Детекция пиков", "🔍 Признаки",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: EDA
# ════════════════════════════════════════════════════════════════════════════

with tab_eda:
    st.subheader(f"Временной ряд: {value_col}")
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
        y=pd.concat([df_plot["rolling_mean"] + df_plot["rolling_std"],
                     (df_plot["rolling_mean"] - df_plot["rolling_std"])[::-1]]),
        fill="toself", fillcolor="rgba(231,76,60,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="±1σ",
    ))
    fig.update_layout(height=380, xaxis_title="Время", yaxis_title=value_col,
                      legend=dict(orientation="h", y=1.1), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    dl(df_plot[["ds", "y", "rolling_mean", "rolling_std"]].round(4),
       "eda_timeseries.csv", "⬇ Временной ряд + скользящее среднее")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Распределение")
        fig2 = px.histogram(df, x="y", nbins=60, color_discrete_sequence=["#3498db"])
        fig2.add_vline(x=df["y"].mean(), line_dash="dash", line_color="red",
                       annotation_text=f"Mean={df['y'].mean():.1f}")
        fig2.add_vline(x=df["y"].median(), line_dash="dot", line_color="orange",
                       annotation_text=f"Median={df['y'].median():.1f}")
        fig2.update_layout(height=300, xaxis_title=value_col, yaxis_title="Частота", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        dl(df[["y"]].rename(columns={"y": value_col}), "eda_distribution.csv", "⬇ Распределение")

    with c2:
        st.subheader("Перцентили")
        quantiles = [50, 75, 90, 95, 99]
        q_vals = [np.percentile(df["y"].dropna(), q) for q in quantiles]
        fig3 = px.bar(x=q_vals, y=[f"P{q}" for q in quantiles], orientation="h",
                      color_discrete_sequence=["#3498db"], text=[f"{v:.1f}" for v in q_vals])
        fig3.update_layout(height=300, xaxis_title=value_col, yaxis_title="", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        dl(pd.DataFrame({"Перцентиль": [f"P{q}" for q in quantiles],
                          value_col: [round(v, 4) for v in q_vals]}),
           "eda_percentiles.csv", "⬇ Перцентили")

    st.subheader("Сезонность")
    df_s = df.assign(hour=df["ds"].dt.hour, day_of_week=df["ds"].dt.dayofweek)

    c3, c4 = st.columns(2)
    with c3:
        hourly = df_s.groupby("hour")["y"].mean().reset_index()
        fig4 = px.line(hourly, x="hour", y="y", markers=True, color_discrete_sequence=["#3498db"])
        fig4.update_layout(title="Суточный профиль (среднее)", xaxis_title="Час суток",
                           yaxis_title=value_col, height=280)
        st.plotly_chart(fig4, use_container_width=True)
        dl(hourly.rename(columns={"hour": "Час", "y": value_col}).round(4),
           "eda_hourly_profile.csv", "⬇ Суточный профиль")

    with c4:
        daily = df_s.groupby("day_of_week")["y"].mean().reset_index()
        daily["day_name"] = daily["day_of_week"].map(lambda i: DAYS_LABELS[i])
        fig5 = px.bar(daily, x="day_name", y="y", color_discrete_sequence=["#3498db"])
        fig5.update_traces(marker_color=["#e74c3c" if i >= 5 else "#3498db" for i in daily["day_of_week"]])
        fig5.update_layout(title="Средняя нагрузка по дням недели",
                           xaxis_title="", yaxis_title=value_col, height=280)
        st.plotly_chart(fig5, use_container_width=True)
        dl(daily[["day_name", "y"]].rename(columns={"day_name": "День", "y": value_col}).round(4),
           "eda_daily_profile.csv", "⬇ Дневной профиль")

    pivot = df_s.groupby(["day_of_week", "hour"])["y"].mean().unstack(fill_value=0)
    pivot.index = [DAYS_LABELS[i] for i in pivot.index]
    fig6 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                     labels=dict(x="Час", y="День", color=value_col), aspect="auto")
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
    c1.metric("Train",      f"{len(train):,} точек", f"{len(train)/n:.0%}")
    c2.metric("Validation", f"{len(val):,} точек",   f"{len(val)/n:.0%}")
    c3.metric("Test",       f"{len(test):,} точек",  f"{len(test)/n:.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train",
                             line=dict(color="#3498db", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=val["ds"],   y=val["y"],   name="Validation",
                             line=dict(color="#f39c12", width=1.2)))
    fig.add_trace(go.Scatter(x=test["ds"],  y=test["y"],  name="Test",
                             line=dict(color="#27ae60", width=1.5)))
    fig.add_vrect(x0=val["ds"].iloc[0],  x1=val["ds"].iloc[-1],
                  fillcolor="orange", opacity=0.07, line_width=0,
                  annotation_text="Val", annotation_position="top left")
    fig.add_vrect(x0=test["ds"].iloc[0], x1=test["ds"].iloc[-1],
                  fillcolor="green", opacity=0.08, line_width=0,
                  annotation_text="Test", annotation_position="top left")
    fig.update_layout(height=420, xaxis_title="Время", yaxis_title=value_col,
                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    split_df = pd.concat([train.assign(split="train"), val.assign(split="val"),
                           test.assign(split="test")])\
                 .rename(columns={"ds": "timestamp", "y": value_col})
    dl(split_df, "split_data.csv", "⬇ Данные с разбиением")

# ════════════════════════════════════════════════════════════════════════════
# ОБУЧЕНИЕ МОДЕЛЕЙ (делегировано в ModelComparison)
# ════════════════════════════════════════════════════════════════════════════

def run_training():
    active_exog = [c for c in config.EXOG_COLS if c in df.columns]
    comparator = ModelComparison(model_save_dir=SAVE_DIR)
    comparator._builder = FeatureBuilder(exog_cols=active_exog)

    comparator.run(train, val, test, include_prophet=use_prophet, include_lstm=use_lstm)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        comparator._builder.transform_splits(train, val, test)

    diag = comparator._builder.diagnostics()
    st.info(
        f"FeatureBuilder построил **{diag['n_features']}** признаков "
        f"(шаг: {diag['step_minutes']:.1f} мин). "
        + (f"⚠️ Пропущены лаги: {diag['skipped_lags_h']}ч." if diag["skipped_lags_h"] else "Все лаги активны.")
    )

    lower_ci = upper_ci = None
    if use_xgb and use_ci and "XGBoost" in comparator.models_:
        lower_ci, upper_ci = get_confidence_interval(
            X_train, y_train, X_val, y_val, X_test, save_dir=SAVE_DIR,
        )

    imp_df = None
    if "XGBoost" in comparator.models_:
        imp_df = feature_importance(comparator.models_["XGBoost"],
                                    feature_names=list(X_train.columns))

    novelty_detector = novelty_threshold = novelty_test_error = None
    novelty_meta = os.path.join(SAVE_DIR, "novelty_meta.pkl")
    if os.path.exists(novelty_meta):
        try:
            from evaluation.novelty_detector import LSTMNoveltyDetector
            nd = LSTMNoveltyDetector.load(SAVE_DIR)
            novelty_detector, novelty_threshold = nd, nd.threshold_
            novelty_test_error = nd.reconstruction_error(test)
        except Exception:
            pass
    else:
        try:
            from evaluation.novelty_detector import LSTMNoveltyDetector
            nd = LSTMNoveltyDetector(window=config.NOVELTY_WINDOW, epochs=config.NOVELTY_EPOCHS,
                                     threshold_q=config.NOVELTY_THRESHOLD_Q)
            nd.fit(train)
            nd.save(SAVE_DIR)
            novelty_detector, novelty_threshold = nd, nd.threshold_
            novelty_test_error = nd.reconstruction_error(test)
        except Exception:
            pass

    return {
        "results": comparator.results_,
        "models":  comparator.models_,
        "preds":   comparator.predictions_,
        "y_test":  y_test,
        "X_train": X_train,
        "lower":   lower_ci,
        "upper":   upper_ci,
        "imp_df":  imp_df,
        "builder": comparator._builder,
        "fb_diagnostics":      diag,
        "novelty_detector":    novelty_detector,
        "novelty_threshold":   novelty_threshold,
        "novelty_test_error":  novelty_test_error,
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
        best = min(results, key=lambda k: results[k]["MAE"])

        diag = trained.get("fb_diagnostics", {})
        if diag:
            with st.expander("🔍 Диагностика FeatureBuilder", expanded=False):
                st.markdown(
                    f"- **Признаков:** {diag.get('n_features', 0)}  \n"
                    f"- **Шаг данных:** {diag.get('step_minutes', 0):.1f} мин  \n"
                    f"- **Строк:** {diag.get('n_rows', 0)}  \n"
                    f"- **Пропущенные лаги:** {diag.get('skipped_lags_h', []) or 'нет'}"
                )
                st.code(", ".join(diag.get("feature_names", [])), language="text")

        # Диагностика переобучения из ModelComparison
        overfit_data = [(n, m["overfit_ratio"]) for n, m in results.items() if "overfit_ratio" in m]
        if overfit_data:
            with st.expander("📊 Диагностика переобучения (train-val gap)", expanded=False):
                for name, ratio in overfit_data:
                    icon = "⚠️" if ratio > 0.3 else "✅"
                    st.markdown(f"{icon} **{name}**: train-val gap = `{ratio*100:.1f}%` (порог 30%)")

        st.subheader("Метрики качества")
        any_unreliable = any(not m.get("MAPE_reliable", True) for m in results.values())
        if any_unreliable:
            st.info("ℹ️ MAPE **ненадёжна** (значения близки к нулю). Ориентируйтесь на **R²**, **SMAPE** и **MAE**.")

        rows = []
        for name, m in results.items():
            mape_val = m.get("MAPE", float("nan"))
            rows.append({
                "Модель":   f"{'★ ' if name == best else ''}{name}",
                "MAE":      round(m["MAE"], 3),
                "RMSE":     round(m["RMSE"], 3),
                "R²":       round(m.get("R2", float("nan")), 3),
                "SMAPE, %": round(m.get("SMAPE", float("nan")), 2),
                "MAPE, %":  f"{mape_val:.2f}" if m.get("MAPE_reliable", True) else f"({mape_val:.1f})*",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Модель"), use_container_width=True)
        if any_unreliable:
            st.caption("(*) — MAPE рассчитана, но ненадёжна")

        model_names = list(results.keys())
        colors = ["#e74c3c" if n == best else "#3498db" for n in model_names]
        fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "MAPE (%)"])
        for i, metric in enumerate(["MAE", "RMSE", "MAPE"], 1):
            vals = [results[m][metric] for m in model_names]
            fig.add_trace(go.Bar(x=model_names, y=vals, marker_color=colors,
                                 text=[f"{v:.2f}" for v in vals], textposition="outside",
                                 showlegend=False), row=1, col=i)
        fig.update_layout(height=350, title_text="Сравнение метрик (меньше = лучше)")
        st.plotly_chart(fig, use_container_width=True)
        dl(pd.DataFrame(rows), "metrics_comparison.csv", "⬇ Метрики сравнения")

        best_r2 = results[best].get("R2", float("nan"))
        msg = (f"R² = {best_r2:.3f}, " if not np.isnan(best_r2) else "") + \
              f"MAE = {results[best]['MAE']:.3f}, SMAPE = {results[best].get('SMAPE', float('nan')):.2f}%"
        st.success(f"🏆 Лучшая модель: **{best}** — {msg}")

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

        zoom    = st.toggle("Только тест-период", value=True)
        show_ci = st.checkbox("Показать доверительный интервал (XGBoost)", value=True)

        model_colors = {"XGBoost": "#e74c3c", "Prophet": "#e67e22", "LSTM": "#9b59b6"}
        fig = go.Figure()

        if not zoom:
            fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train",
                                     line=dict(color="#3498db", width=0.8), opacity=0.5))
            fig.add_trace(go.Scatter(x=val["ds"], y=val["y"], name="Validation",
                                     line=dict(color="#f39c12", width=0.8), opacity=0.7))

        n_test  = min(len(test["ds"]), len(y_test))
        test_ds = test["ds"].values[:n_test]

        if show_ci and lower_ci is not None and upper_ci is not None and "XGBoost" in preds_all:
            n_ci = min(n_test, len(lower_ci), len(upper_ci))
            coverage = np.mean((y_test.values[:n_ci] >= lower_ci[:n_ci]) &
                                (y_test.values[:n_ci] <= upper_ci[:n_ci]))
            fig.add_trace(go.Scatter(
                x=np.concatenate([test_ds[:n_ci], test_ds[:n_ci][::-1]]),
                y=np.concatenate([upper_ci[:n_ci], lower_ci[:n_ci][::-1]]),
                fill="toself", fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"ДИ 80% XGBoost (покрытие {coverage:.0%})",
            ))

        for name, preds in preds_all.items():
            n_p = min(n_test, len(preds))
            fig.add_trace(go.Scatter(x=test_ds[:n_p], y=preds[:n_p],
                                     name=f"{name} (прогноз)",
                                     line=dict(color=model_colors.get(name, "#333"), width=1.8, dash="dash")))

        fig.add_trace(go.Scatter(x=test_ds, y=y_test.values[:n_test], name="Факт",
                                 line=dict(color="#27ae60", width=1.5), opacity=0.7))
        fig.update_layout(height=480, xaxis_title="Время", yaxis_title=value_col,
                          hovermode="x unified", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

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

        if "XGBoost" in preds_all:
            st.subheader("Абсолютная ошибка XGBoost (первые 48 точек)")
            n_show = min(48, n_test, len(preds_all["XGBoost"]))
            err_df = pd.DataFrame({
                "Время":    test["ds"].values[:n_show],
                "Факт":     y_test.values[:n_show].round(2),
                "Прогноз":  preds_all["XGBoost"][:n_show].round(2),
                "Ошибка":   np.abs(y_test.values[:n_show] - preds_all["XGBoost"][:n_show]).round(2),
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
        upper_ci  = trained.get("upper")

        if use_upper_for_scaling and upper_ci is not None and len(upper_ci) > 0:
            preds_for_decision = np.asarray(upper_ci)
            decision_label = "Верхняя граница (P90)"
            st.success("🛡️ **Проактивный режим:** решения принимаются по **верхней границе ДИ** (P90).")
        else:
            preds_for_decision = np.asarray(preds_xgb)
            decision_label = "Точечный прогноз"
            if use_upper_for_scaling and upper_ci is None:
                st.warning("⚠️ ДИ недоступен — детекция по точечному прогнозу. Включите «ДИ XGBoost».")

        detector = PeakDetector(
            target_rps_per_replica=float(train["y"].max()) / config.MAX_REPLICAS,
            min_replicas=config.MIN_REPLICAS,
            max_replicas=config.MAX_REPLICAS,
            warning_ratio=config.ALERT_WARNING_RATIO,
            critical_ratio=config.ALERT_CRITICAL_RATIO,
        )
        detector.fit(train["y"])

        n_test = min(len(test["ds"]), len(y_test), len(preds_for_decision))
        predicted_series = pd.Series(preds_for_decision[:n_test], index=y_test.index[:n_test])
        events_df = detector.detect_series(
            y_test.iloc[:n_test], predicted_series,
            recompute_every=getattr(config, "PEAK_RECOMPUTE_EVERY", None),
        )
        summary = detector.summary(events_df)
        st.caption(f"Сигнал для детекции: **{decision_label}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Порог",       f"{summary['threshold']:.1f}")
        c2.metric("Пиков",       f"{summary['peaks_detected']}")
        c3.metric("Доля пиков",  f"{summary['peak_ratio_pct']}%")
        c4.metric("Critical",    summary["severity_counts"].get("critical", 0))

        sev_color = {"ok": "#27ae60", "info": "#3498db", "warning": "#f39c12",
                     "critical": "#e74c3c", "exceeded": "#e67e22"}
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            subplot_titles=["Прогноз vs факт + пороги", "Severity"])
        fig.add_trace(go.Scatter(x=events_df["timestamp"], y=events_df["rps"],
                                 name="Факт", line=dict(color="#27ae60", width=1.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=events_df["timestamp"], y=events_df["predicted"],
                                 name=f"Сигнал: {decision_label}",
                                 line=dict(color="#e74c3c", width=1.5, dash="dash")), row=1, col=1)
        if use_upper_for_scaling and upper_ci is not None:
            n_show = min(n_test, len(preds_xgb))
            fig.add_trace(go.Scatter(x=events_df["timestamp"].iloc[:n_show],
                                     y=np.asarray(preds_xgb)[:n_show],
                                     name="Точечный прогноз (для сравнения)",
                                     line=dict(color="#9b59b6", width=1, dash="dot")), row=1, col=1)
        fig.add_hline(y=detector.threshold, line_dash="dot", line_color="black",
                      annotation_text=f"Порог ({detector.threshold:.0f})", row=1, col=1)
        fig.add_hline(y=detector.threshold * config.ALERT_WARNING_RATIO,
                      line_dash="dot", line_color="#f39c12", annotation_text="Warning", row=1, col=1)
        sev_map = {"ok": 0, "info": 1, "warning": 2, "critical": 3, "exceeded": 4}
        fig.add_trace(go.Scatter(
            x=events_df["timestamp"], y=events_df["severity"].map(sev_map), mode="markers",
            marker=dict(color=[sev_color[s] for s in events_df["severity"]], size=4),
            name="Severity",
        ), row=2, col=1)
        fig.update_yaxes(tickvals=[0, 1, 2, 3, 4],
                         ticktext=["ok", "info", "warning", "critical", "exceeded"], row=2, col=1)
        fig.update_layout(height=550, hovermode="x unified", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        dl(events_df.round(4), "peak_events_all.csv", "⬇ Все события (факт + прогноз + severity)")

        peaks = events_df[events_df["is_peak"]].copy()
        if len(peaks) > 0:
            st.subheader(f"Обнаруженные пики ({len(peaks)} шт.)")
            peaks_show = peaks[["timestamp", "rps", "predicted", "severity", "recommended_replicas"]].copy()
            peaks_show.columns = ["Время", "Факт", "Прогноз", "Severity", "Реплик"]
            peaks_show[["Факт", "Прогноз"]] = peaks_show[["Факт", "Прогноз"]].round(1)
            st.dataframe(peaks_show.reset_index(drop=True), use_container_width=True, height=300)
            dl(peaks_show.reset_index(drop=True), "peak_events_only.csv", "⬇ Только пики")

        st.subheader("Рекомендуемое число реплик (по прогнозу)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=events_df["timestamp"], y=events_df["recommended_replicas"],
                                  mode="lines", line=dict(color="#3498db", width=2, shape="hv"),
                                  fill="tozeroy", fillcolor="rgba(52,152,219,0.15)", name="Реплик"))
        fig2.add_hline(y=config.MIN_REPLICAS, line_dash="dash", line_color="green",
                       annotation_text=f"Min={config.MIN_REPLICAS}")
        fig2.add_hline(y=config.MAX_REPLICAS, line_dash="dash", line_color="red",
                       annotation_text=f"Max={config.MAX_REPLICAS}")
        fig2.update_layout(height=280, yaxis_title="Реплик",
                           yaxis=dict(dtick=1, range=[0, config.MAX_REPLICAS + 1]))
        st.plotly_chart(fig2, use_container_width=True)
        dl(events_df[["timestamp", "predicted", "recommended_replicas"]].round(4),
           "scaling_recommendations.csv", "⬇ Рекомендации по репликам")

        nd = trained.get("novelty_detector")
        if nd is not None:
            st.subheader("Детектор новизны (LSTM Autoencoder)")
            err, thr = trained.get("novelty_test_error", 0.0), trained.get("novelty_threshold", 0.0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Ошибка восстановления",  f"{err:.4f}")
            c2.metric("Порог (95-й перцентиль)", f"{thr:.4f}")
            c3.metric("Паттерн тестового окна",  "⚠️ Новый" if err > thr else "✅ Знакомый")
            st.caption("Высокая ошибка восстановления означает, что паттерн нагрузки "
                       "в тестовом окне отличается от наблюдавшихся при обучении.")
        else:
            st.info("Детектор новизны недоступен. Запустите `python main.py train --synthetic` "
                    "или установите TensorFlow.")

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

        imp_plot = imp_df.copy()
        imp_plot["importance_pct"] = (imp_df["importance"] / imp_df["importance"].sum() * 100).round(2)

        fig = px.bar(imp_plot.sort_values("importance"), x="importance_pct", y="feature",
                     orientation="h", color="importance_pct", color_continuous_scale="RdYlGn",
                     text="importance_pct")
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

        fig2 = px.bar(x=corr.values, y=corr.index, orientation="h",
                      color=corr.values, color_continuous_scale="RdBu", color_continuous_midpoint=0)
        fig2.update_layout(height=380, xaxis_title="Корреляция Пирсона", coloraxis_showscale=False)
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
