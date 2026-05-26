# """
# Генерация синтетических данных на основе Azure Functions Dataset 2019.
# Шаг 1: извлекаем реальные RPS-ряды топ-N приложений из архива (чанками).
# Шаг 2: обучаем PARSynthesizer (SDV) и генерируем расширенный датасет.
# Шаг 3: добавляем экзогенные признаки и сохраняем результат.
# """

# import os
# import math
# import tarfile
# import numpy as np
# import pandas as pd
# from sdv.metadata import Metadata
# from sdv.sequential import PARSynthesizer

# ARCHIVE      = r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data\azurefunctions-dataset2019.tar.xz"
# OUT_CSV      = r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data\synthetic_azure.csv"
# MODEL_PATH   = r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data\par_model.pkl"
# TOP_N        = 5      # приложений
# EPOCHS       = 30
# GEN_DAYS     = 30     # итоговый объём на приложение
# SEQ_DAYS     = 7      # длина одного запроса к PAR (не более длины обучающего ряда)
# CHUNK        = 5_000

# # ── 1. Чтение архива чанками → агрегация по HashApp ──────────────────────
# def _load_archive(archive: str, top_apps: set | None = None) -> pd.DataFrame:
#     parts = []
#     with tarfile.open(archive, "r:xz") as tar:
#         members = sorted(
#             [m for m in tar.getmembers()
#              if m.isfile() and "invocations_per_function_md" in m.name],
#             key=lambda m: m.name,
#         )
#         for member in members:
#             day = int(member.name.split(".d")[-1].split(".")[0])
#             agg_chunks = []
#             for chunk in pd.read_csv(tar.extractfile(member), chunksize=CHUNK, low_memory=False):
#                 mc = [c for c in chunk.columns if c.isdigit()]
#                 if top_apps is not None:
#                     chunk = chunk[chunk["HashApp"].isin(top_apps)]
#                 if len(chunk):
#                     agg_chunks.append(chunk.groupby("HashApp")[mc].sum())
#             if not agg_chunks:
#                 continue
#             agg = pd.concat(agg_chunks).groupby(level=0).sum()
#             mc = [c for c in agg.columns if c.isdigit()]
#             melted = agg.reset_index().melt(id_vars="HashApp", value_vars=mc,
#                                             var_name="minute", value_name="rps")
#             melted["ds"] = pd.Timestamp("2019-01-01") + pd.to_timedelta(
#                 (day - 1) * 1440 + melted["minute"].astype(int), unit="min"
#             )
#             melted["rps"] /= 60
#             parts.append(melted[["HashApp", "ds", "rps"]])
#     return pd.concat(parts, ignore_index=True).sort_values(["HashApp", "ds"]).reset_index(drop=True)

# print("Проход 1: определяем топ приложений...")
# totals = pd.Series(dtype=float)
# with tarfile.open(ARCHIVE, "r:xz") as tar:
#     for m in tar.getmembers():
#         if m.isfile() and "invocations_per_function_md" in m.name:
#             for chunk in pd.read_csv(tar.extractfile(m), chunksize=CHUNK, low_memory=False):
#                 mc = [c for c in chunk.columns if c.isdigit()]
#                 totals = totals.add(chunk.groupby("HashApp")[mc].sum().sum(axis=1), fill_value=0)
# top_apps = set(totals.nlargest(TOP_N).index)

# print("Проход 2: читаем данные топ приложений...")
# df = _load_archive(ARCHIVE, top_apps)
# df = df.rename(columns={"HashApp": "app_id"})
# print(f"  Загружено {len(df):,} строк, {df['app_id'].nunique()} приложений")

# # ── 2. Обучение PARSynthesizer ────────────────────────────────────────────
# print("Обучение PARSynthesizer...")
# # detect_from_dataframe — classmethod, возвращает новый объект Metadata
# metadata = Metadata.detect_from_dataframe(df)
# metadata.update_column("ds",     sdtype="datetime", datetime_format="%Y-%m-%d %H:%M:%S")
# metadata.update_column("app_id", sdtype="id")
# metadata.update_column("rps",    sdtype="numerical", computer_representation="Float")
# metadata.set_sequence_key("app_id")
# metadata.set_sequence_index("ds")

# syn = PARSynthesizer(metadata, epochs=EPOCHS, verbose=True)
# if os.path.exists(MODEL_PATH):
#     print(f"Загружаем сохранённую модель из {MODEL_PATH}...")
#     syn = PARSynthesizer.load(MODEL_PATH)
# else:
#     syn.fit(df)
#     syn.save(MODEL_PATH)
#     print(f"Модель сохранена → {MODEL_PATH}")

# # ── 3. Генерация расширенного ряда (короткими блоками по SEQ_DAYS) ────────
# print("Генерация синтетических данных...")
# n_batches = math.ceil(GEN_DAYS / SEQ_DAYS)
# parts_syn = []
# for batch_idx in range(n_batches):
#     print(f"  Батч {batch_idx + 1}/{n_batches} ({SEQ_DAYS} дней × {TOP_N} приложений)...")
#     chunk_syn = syn.sample(
#         num_sequences=TOP_N,
#         sequence_length=SEQ_DAYS * 1440,
#     )
#     chunk_syn["_batch"] = batch_idx
#     parts_syn.append(chunk_syn)

# synthetic = pd.concat(parts_syn, ignore_index=True)
# synthetic.drop(columns=["_batch"], inplace=True)

# # ── 4. Добавляем экзогенные признаки ────────────────────────────────────
# synthetic["ds"] = pd.to_datetime(synthetic["ds"])
# synthetic["is_weekend"]  = synthetic["ds"].dt.dayofweek.isin([5, 6]).astype(int)
# synthetic["hour"]        = synthetic["ds"].dt.hour
# synthetic["day_of_week"] = synthetic["ds"].dt.dayofweek

# # Плавный тренд роста нагрузки (эмуляция Data Drift)
# n = len(synthetic)
# synthetic["month_trend"] = np.linspace(1.0, 1.3, n)
# synthetic["rps"] *= synthetic["month_trend"]
# synthetic["rps"] *= np.where(synthetic["is_weekend"], 0.75, 1.0)
# synthetic["rps"] = synthetic["rps"].clip(lower=0)

# synthetic.to_csv(OUT_CSV, index=False)
# print(f"Готово: {len(synthetic):,} строк → {OUT_CSV}")
# print(synthetic.head())

import tarfile
import pandas as pd

ARCHIVE = r"C:\Users\qwesd\Desktop\Obshaya\Vuz 4 kurs\Diplome\proactive-scaler\data\azurefunctions-dataset2019.tar.xz"

with tarfile.open(ARCHIVE, "r:xz") as tar:
    for member in tar.getmembers():
        if member.isfile() and "invocations_per_function_md" in member.name:
            df = pd.read_csv(tar.extractfile(member))
            break

# Информация о структуре
print("Таблица 1 — Структура исходных данных\n")
print(f"{'Поле':<20} {'Тип':<15} {'Описание'}")
print("-" * 70)
print(f"{'HashOwner':<20} {'строка':<15} {'Хеш владельца приложения'}")
print(f"{'HashApp':<20} {'строка':<15} {'Хеш приложения (идентификатор)'}")
print(f"{'HashFunction':<20} {'строка':<15} {'Хеш функции'}")
print(f"{'Trigger':<20} {'строка':<15} {'Тип триггера (http, timer, ...)'}")
print(f"{'0-1439':<20} {'целое':<15} {'Счётчик вызовов за минуту (всего 1440 колонок)'}")

print(f"\nВсего строк (уникальных функций): {len(df):,}")
print(f"Всего колонок: {df.shape[1]} (4 атрибута + 1440 минут)")