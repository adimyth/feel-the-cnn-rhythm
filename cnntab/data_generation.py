import random
from multiprocessing import Pool
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns


def df_to_heatmap(
    data: pd.DataFrame,
    idx: int = None,
    min_offset: int = 120,
    offset: int = 0,
):
    if idx is None:
        diag = data.iloc[min_offset:, :]
        idx = diag[diag.worked].index[offset]
    timestamp = data.iloc[idx, :].timestamp
    diag = data.iloc[idx - min_offset : idx, :]
    label = diag.worked.sum()
    subset = diag.worked.values.reshape((min_offset // 24, 24))
    return (subset, label, timestamp, diag)


def create_heatmap(subset):
    fig, _ = plt.subplots(figsize=(10, 5))
    _ = sns.heatmap(subset, linewidths=1, cmap="Greens", linecolor="white", cbar=False)
    plt.axis("off")
    return fig


def generate_parallel_heatmaps(df):
    df = df[1]
    df = df.reset_index(drop=True)
    for idx in df[df.worked].index:
        if idx <= 120:
            continue
        subset, label, timestamp, _ = df_to_heatmap(df, idx=idx)
        if random.random() > 0.1:
            continue

        h = create_heatmap(subset)
        pathcomp_1 = str(df.worker.unique()[0])
        pathcomp_2 = str(timestamp.timestamp())
        path_str = f"{pathcomp_1}_{pathcomp_2}"
        path_hash = hashlib.md5(path_str.encode("utf-8")).hexdigest()
        h.savefig(
            Path("data/heatmaps") / f"{label}_{path_hash}.png",
            bbox_inches="tight",
        )
        plt.close(h)


def generate_all_heatmaps(input_path: str):
    if Path(input_path).exists():
        data = pd.read_csv(str(input_path))
        data.timestamp = pd.to_datetime(data.timestamp)
    else:
        raise ValueError(f"FTR data at path {input_path} doesn't exist!")

    pool = Pool()
    for _ in tqdm(
        pool.imap_unordered(generate_parallel_heatmaps, data.groupby("worker")),
        total=data.worker.nunique(),
    ):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    generate_all_heatmaps(input_path="data/final.csv")
