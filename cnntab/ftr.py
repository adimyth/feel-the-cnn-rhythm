import gc
from datetime import date, timedelta
from pathlib import Path
from typing import List, Union, Optional
import datetime as dt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
import random
from heatmap import Heatmap
import matplotlib.pyplot as plt
import matplotlib as mpl


def extract_one_worker(file_path: Union[str, Path], offset: int = 0):
    if Path(file_path).exists():
        data = pd.read_csv(str(file_path))
        data.timestamp = pd.to_datetime(data.timestamp)
    else:
        raise ValueError(f"FTR data at path {file_path} doesn't exist!")
    for idx, grp in enumerate(data.groupby("worker")):
        if idx == offset:
            return grp[1]
    raise ValueError(f"Incorrect offset requested: {offset}. Max possible: {idx}")


def df_to_heatmap(
    data: pd.DataFrame,
    idx: int = None,
    min_offset: int = 120,
    offset: int = 0,
):
    if idx is None:
        diag = data.iloc[min_offset:, :]
        idx = diag[diag.worked == True].index[offset]
    label = data.iloc[idx, :].incident
    timestamp = data.iloc[idx, :].timestamp
    diag = data.iloc[idx - 120 : idx, :]
    subset = diag.worked.values.reshape((5, 24))
    return (subset, label, timestamp, diag)


def df_to_heatmap_v2(data: pd.DataFrame, min_offset: int = 120, offset: int = 0):
    diag = data.iloc[min_offset:, :]
    idx = diag[diag.worked == True].index[offset]
    label = data.iloc[idx, :].incident
    timestamp = data.iloc[idx, :].timestamp

    start = data.timestamp[idx] - dt.timedelta(days=4)
    start = pd.Series(start).dt.floor("D").dt.strftime("%Y-%m-%d %H:%M:%S")
    end = data.timestamp[idx]
    end = pd.Series(end)
    # end = pd.Series(end).dt.ceil("D").dt.strftime("%Y-%m-%d %H:%M:%S")

    diag = data[data.timestamp.isin(pd.date_range(start[0], end[0], freq="H"))]

    subset = diag.worked.values
    subset = np.append(subset, np.repeat(False, 120 - len(subset)))
    subset = subset.reshape((5, 24))
    return (subset, label, timestamp, diag)


def generate_one_binary_mask(subset):
    # Setup
    cmap = mpl.colors.ListedColormap(["w", "g"])
    bounds = [0.0, 0.5, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xticks(range(0, 24))
    plt.yticks(range(0, 5))
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.imshow(subset, interpolation="none", cmap=cmap, norm=norm)
    return fig


def generate_all_heatmaps(input_path: str, output_path: str = "data/"):
    if Path(input_path).exists():
        data = pd.read_csv(str(input_path))
        data.timestamp = pd.to_datetime(data.timestamp)
    else:
        raise ValueError(f"FTR data at path {input_path} doesn't exist!")
    for idx, df in tqdm(data.groupby("worker")):
        # if df.shape[0] < 120:
        #     continue
        df = df.reset_index(drop=True)

        for idx in df[df.worked == True].index:
            if idx <= 120:
                continue
            subset, label, timestamp, diag = df_to_heatmap(df, idx=idx)
            if label == False:
                if random.random() > 0.1:
                    continue

            h = generate_one_binary_mask(subset)
            h.savefig(
                Path(output_path)
                / f"{int(label)}_{df.worker.unique()[0]}_{timestamp.strftime('%Y_%m_%d_%H_%M_%S')}.png"
            )
            plt.close(h)


class FTR(BaseModel):
    file_path: Union[str, Path]
    data: Optional[pd.DataFrame]
    employee_records: Optional[pd.DataFrame]

    class Config:
        arbitrary_types_allowed = True

    def load_data(self) -> pd.DataFrame:
        if Path(self.file_path).exists():
            df = pd.read_csv(str(self.file_path))
        else:
            raise ValueError(f"FTR data at path {self.file_path} doesn't exist!")

        # considering only employees which have faced an incident
        emps = df.loc[df["incident"] == True]["EmpNo_Anon"].tolist()
        df = df.loc[df["EmpNo_Anon"].isin(emps)]
        df = df[["EmpNo_Anon", "Work_DateTime", "incident"]]
        df["Work_DateTime"] = pd.to_datetime(df["Work_DateTime"])

        self.data = df
        return self

    def get_employee_record(self, sample: bool = True):
        self.employee_records = pd.DataFrame()
        for master_idx, df in tqdm(self.data.groupby("EmpNo_Anon")):
            if sample == True:
                if random.random() > 0.01:
                    continue
            new_df = pd.DataFrame(
                False,
                index=pd.date_range(
                    df.Work_DateTime.min(), df.Work_DateTime.max(), freq="H"
                ),
                columns=["worked"],
            )
            new_df.loc[:, "incident"] = False
            df = df.reset_index(drop=True)
            for idx, item in df.iterrows():
                new_df.loc[df.iloc[idx].Work_DateTime, "worked"] = True
                new_df.loc[df.iloc[idx].Work_DateTime, "incident"] = df.iloc[
                    idx
                ].incident
            new_df.loc[:, "worker"] = master_idx
            self.employee_records = pd.concat([self.employee_records, new_df])

        self.employee_records.index = self.employee_records.index.set_names("timestamp")
        self.employee_records.index = pd.to_datetime(self.employee_records.index)
        self.employee_records = self.employee_records.reset_index(drop=False)
        return self


if __name__ == "__main__":
    # Generate employee history
    # x = (
    #     FTR(file_path="data/public.csv")
    #     .load_data()
    #     .get_employee_record(sample=False)
    #     .employee_records.to_csv("data/final.csv", index=False)
    # )
    # Generate heat maps
    generate_all_heatmaps(input_path="data/final.csv", output_path="data/heatmaps/")
