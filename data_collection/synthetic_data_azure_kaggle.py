"""
Генерация синтетических данных на основе Azure Functions Dataset 2019.
Версия для Kaggle: стриминговое чтение архива (r|xz) без загрузки в память.
"""

import io
import os
import gc
import tarfile
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

ARCHIVE    = "/root/sdv_project/azurefunctions-dataset2019.tar.xz"
OUT_CSV    = "/root/sdv_project/synthetic_azure.csv"
MODEL_PATH = "/root/sdv_project/par_model.pkl"
TOP_N      = 3       # приложений — 3 достаточно для диплома
EPOCHS     = 8       # было 20; 8 даёт приемлемое качество намного быстрее
GEN_DAYS   = 90      # 3 месяца
SEQ_DAYS   = 14      # блоки по 14 дней = длина обучения → естественные переходы
CHUNK      = 10_000  # больший чанк → быстрее читает CSV
MAX_DAYS   = 14       # читаем только первые 5 дней архива для обучения


# ── 1a. Стриминговый проход: определяем топ приложений ───────────────────
print("Проход 1: определяем топ приложений (стриминг)...")
totals = pd.Series(dtype=float)
days_seen = 0

with tarfile.open(ARCHIVE, "r|xz") as tar:
    for member in tar:
        if not (member.isfile() and "invocations_per_function_md" in member.name):
            continue
        if days_seen >= MAX_DAYS:
            break
        days_seen += 1
        f = tar.extractfile(member)
        if f is None:
            continue
        buf = io.BytesIO(f.read())
        for chunk in pd.read_csv(buf, chunksize=CHUNK, low_memory=False):
            mc = [c for c in chunk.columns if c.isdigit()]
            totals = totals.add(
                chunk.groupby("HashApp")[mc].sum().sum(axis=1), fill_value=0
            )
        del buf
        gc.collect()

top_apps = set(totals.nlargest(TOP_N).index)
del totals
gc.collect()
print(f"  Топ приложения: {top_apps}")

# ── 1b. Стриминговый проход: читаем данные топ приложений ────────────────
print("Проход 2: читаем данные топ приложений (стриминг)...")
parts = []
days_seen = 0

with tarfile.open(ARCHIVE, "r|xz") as tar:
    for member in tar:
        if not (member.isfile() and "invocations_per_function_md" in member.name):
            continue
        if days_seen >= MAX_DAYS:
            break
        day = int(member.name.split(".d")[-1].split(".")[0])
        days_seen += 1
        f = tar.extractfile(member)
        if f is None:
            continue

        buf = io.BytesIO(f.read())
        agg_chunks = []
        for chunk in pd.read_csv(buf, chunksize=CHUNK, low_memory=False):
            mc = [c for c in chunk.columns if c.isdigit()]
            chunk = chunk[chunk["HashApp"].isin(top_apps)]
            if len(chunk):
                agg_chunks.append(chunk.groupby("HashApp")[mc].sum())

        if not agg_chunks:
            continue

        agg = pd.concat(agg_chunks).groupby(level=0).sum()
        mc = [c for c in agg.columns if c.isdigit()]
        melted = agg.reset_index().melt(
            id_vars="HashApp", value_vars=mc,
            var_name="minute", value_name="rps"
        )
        melted["ds"] = pd.Timestamp("2019-01-01") + pd.to_timedelta(
            (day - 1) * 1440 + melted["minute"].astype(int), unit="min"
        )
        melted["rps"] /= 60
        parts.append(melted[["HashApp", "ds", "rps"]])
        del buf, agg, melted, agg_chunks
        gc.collect()
        print(f"  День {day} загружен")

df = pd.concat(parts, ignore_index=True).sort_values(["HashApp", "ds"]).reset_index(drop=True)
df = df.rename(columns={"HashApp": "app_id"})
del parts
gc.collect()
print(f"  Загружено {len(df):,} строк, {df['app_id'].nunique()} приложений")

# ── 2. Обучение PARSynthesizer ────────────────────────────────────────────
print("Обучение PARSynthesizer...")
metadata = Metadata.detect_from_dataframe(df)
metadata.update_column("ds",     sdtype="datetime", datetime_format="%Y-%m-%d %H:%M:%S")
metadata.update_column("app_id", sdtype="id")
metadata.update_column("rps",    sdtype="numerical", computer_representation="Float")
metadata.set_sequence_key("app_id")
metadata.set_sequence_index("ds")

if os.path.exists(MODEL_PATH):
    print(f"Загружаем сохранённую модель из {MODEL_PATH}...")
    syn = PARSynthesizer.load(MODEL_PATH)
else:
    syn = PARSynthesizer(metadata, epochs=EPOCHS, verbose=True)
    syn.fit(df)
    syn.save(MODEL_PATH)
    print(f"Модель сохранена → {MODEL_PATH}")

del df
gc.collect()

# ── 3. Генерация блоками по SEQ_DAYS дней и склейка ──────────────────────
# Блоки по 14 дней совпадают с длиной обучения PAR → внутри блока
# переходы естественные. На стыке блоков применяется линейное сглаживание
# (60-минутное окно), чтобы убрать ступенчатые артефакты.
SMOOTH_MIN = 60   # минут сглаживания на границе блоков
n_blocks   = -(-GEN_DAYS // SEQ_DAYS)   # ceil division
print(f"Генерация {GEN_DAYS} дней ({n_blocks} блоков по {SEQ_DAYS} дн.)...")
parts_syn = []
start_ts  = pd.Timestamp("2023-01-01")
day_cursor = 0

for block_idx in range(n_blocks):
    days_this = min(SEQ_DAYS, GEN_DAYS - day_cursor)
    print(f"  Блок {block_idx + 1}/{n_blocks} (дни {day_cursor+1}–{day_cursor+days_this})...")
    chunk_syn = syn.sample(
        num_sequences=TOP_N,
        sequence_length=days_this * 1440,
    )
    for app_id in chunk_syn["app_id"].unique():
        mask = chunk_syn["app_id"] == app_id
        n_pts = mask.sum()
        chunk_syn.loc[mask, "ds"] = pd.date_range(
            start=start_ts + pd.Timedelta(days=day_cursor),
            periods=n_pts,
            freq="1min",
        )
    parts_syn.append(chunk_syn)
    day_cursor += days_this
    gc.collect()

synthetic = pd.concat(parts_syn, ignore_index=True)
del parts_syn
gc.collect()

# Агрегируем по времени до применения сглаживания
synthetic = (
    synthetic.groupby("ds", as_index=False)["rps"].sum()
    .sort_values("ds").reset_index(drop=True)
)

# Сглаживаем швы между блоками: линейный blend на SMOOTH_MIN минут
block_size = SEQ_DAYS * 1440
for b in range(1, n_blocks):
    seam = b * block_size
    if seam >= len(synthetic) or seam < SMOOTH_MIN:
        continue
    start_i = seam - SMOOTH_MIN
    end_i   = min(seam + SMOOTH_MIN, len(synthetic))
    left_val  = synthetic.loc[start_i, "rps"]
    right_val = synthetic.loc[min(seam, len(synthetic)-1), "rps"]
    weights   = np.linspace(0, 1, end_i - start_i)
    synthetic.loc[start_i:end_i-1, "rps"] = (
        left_val * (1 - weights) + right_val * weights
    )
print(f"  Сглаживание швов: {n_blocks-1} границ × {SMOOTH_MIN} мин")

# ── 4. Добавляем экзогенные признаки ─────────────────────────────────────
import holidays as holidays_lib

synthetic["ds"] = pd.to_datetime(synthetic["ds"])

# --- Базовые календарные ---
synthetic["is_weekend"]  = synthetic["ds"].dt.dayofweek.isin([5, 6]).astype(int)
synthetic["hour"]        = synthetic["ds"].dt.hour
synthetic["day_of_week"] = synthetic["ds"].dt.dayofweek

# --- Российские праздники ---
ru_holidays = holidays_lib.Russia(years=[2023])
synthetic["is_holiday"] = (
    synthetic["ds"].dt.normalize()
    .apply(lambda d: d.date() in ru_holidays)
    .astype(int)
)

# --- Событийные пики ---
# is_campaign (×2.2): 1 раз в train + 1 в test → доказывает обобщение с одного примера
# is_promo    (×1.6): 3 раза в train + 1 в test → доказывает стабильность при повторах
rng = np.random.default_rng(42)

# is_campaign: 1 в train (дни 10–70), 1 в test (день 79)
campaign_train = rng.choice(range(10, 70), size=1, replace=False)
campaign_dates = set()
for d in list(campaign_train) + [79]:
    for offset in range(3):
        campaign_dates.add(
            (pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(d) + offset)).normalize()
        )
synthetic["is_campaign"] = (
    synthetic["ds"].dt.normalize().isin(campaign_dates).astype(int)
)

# is_promo: 3 в train (дни 10–70, не пересекаются с campaign), 1 в test (день 83)
used_days = set(int(d) + o for d in campaign_train for o in range(3))
promo_pool = [d for d in range(10, 70) if not any((d + o) in used_days for o in range(3))]
promo_train = rng.choice(promo_pool, size=3, replace=False)
promo_dates = set()
for d in list(promo_train) + [83]:
    for offset in range(3):
        promo_dates.add(
            (pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(d) + offset)).normalize()
        )
synthetic["is_promo"] = (
    synthetic["ds"].dt.normalize().isin(promo_dates).astype(int)
)

print(f"  Праздничных точек:  {synthetic['is_holiday'].sum():,}")
print(f"  Кампанийных точек:  {synthetic['is_campaign'].sum():,}  (1 train + 1 test)")
print(f"  Промо-точек:        {synthetic['is_promo'].sum():,}    (3 train + 1 test)")

# --- Тренд роста (+30% за 90 дней) ---
n = len(synthetic)
synthetic["month_trend"] = np.linspace(1.0, 1.5, n)

# --- Применяем множители к rps ---
synthetic["rps"] *= synthetic["month_trend"]
synthetic["rps"] *= np.where(synthetic["is_weekend"], 0.75, 1.0)
synthetic["rps"] *= np.where(synthetic["is_holiday"],  1.8, 1.0)   # праздник ×1.8
synthetic["rps"] *= np.where(synthetic["is_campaign"], 2.2, 1.0)   # кампания ×2.2
synthetic["rps"] *= np.where(synthetic["is_promo"],    1.6, 1.0)   # промо ×1.6
synthetic["rps"]  = synthetic["rps"].clip(lower=0)

# Финальная сортировка (агрегация rps уже выполнена выше, до умножения)
synthetic = synthetic.sort_values("ds").reset_index(drop=True)

synthetic.to_csv(OUT_CSV, index=False)
print(f"Готово: {len(synthetic):,} строк → {OUT_CSV}")
print(f"Колонки: {list(synthetic.columns)}")
print(synthetic.head())
