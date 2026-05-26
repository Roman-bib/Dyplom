import tarfile
import pandas as pd
import matplotlib.pyplot as plt

archive_path = "azurefunctions-dataset2019.tar.xz"
TOP_N_APPS = 10
CHUNKSIZE = 5000

def get_members(tar):
    return sorted(
        [m for m in tar.getmembers() if m.isfile() and 'invocations_per_function_md' in m.name],
        key=lambda m: m.name
    )

def read_agg(tar, member, top_apps=None):
    """Читает один файл чанками, агрегирует по HashApp."""
    chunks = []
    for chunk in pd.read_csv(tar.extractfile(member), chunksize=CHUNKSIZE, low_memory=False):
        minute_cols = [c for c in chunk.columns if c.isdigit()]
        if top_apps is not None:
            chunk = chunk[chunk['HashApp'].isin(top_apps)]
        if len(chunk):
            chunks.append(chunk.groupby('HashApp')[minute_cols].sum())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).groupby(level=0).sum()

# Проход 1: найти топ-N приложений
print("Проход 1: подсчёт топ приложений...")
app_totals = pd.Series(dtype=float)
with tarfile.open(archive_path, 'r:xz') as tar:
    for member in get_members(tar):
        print(f"  {member.name}")
        agg = read_agg(tar, member)
        app_totals = app_totals.add(agg.sum(axis=1), fill_value=0)

top_apps = set(app_totals.nlargest(TOP_N_APPS).index)
print(f"Топ-{TOP_N_APPS} выбраны")

# Проход 2: собрать временной ряд
print("Проход 2: сборка временного ряда...")
series_parts = []
with tarfile.open(archive_path, 'r:xz') as tar:
    for member in get_members(tar):
        day_num = int(member.name.split('.d')[-1].split('.')[0])
        agg = read_agg(tar, member, top_apps=top_apps)
        minute_cols = [c for c in agg.columns if c.isdigit()]
        melted = agg.reset_index().melt(id_vars='HashApp', value_vars=minute_cols, var_name='minute', value_name='rps')
        melted['ds'] = pd.Timestamp('2019-01-01') + pd.to_timedelta(
            (day_num - 1) * 1440 + melted['minute'].astype(int), unit='min'
        )
        melted['rps'] /= 60
        series_parts.append(melted[['HashApp', 'ds', 'rps']])

result = pd.concat(series_parts, ignore_index=True).sort_values(['HashApp', 'ds']).reset_index(drop=True)
print(f"Готово: {len(result):,} строк")

fig, ax = plt.subplots(figsize=(16, 5))
for app_id, group in result.groupby('HashApp'):
    ax.plot(group['ds'], group['rps'], linewidth=0.6, label=str(app_id)[:8])
ax.set_title(f'RPS — топ {TOP_N_APPS} приложений Azure Functions 2019')
ax.set_ylabel('RPS')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
