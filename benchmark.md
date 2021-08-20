# Benchmarking Runtime of FTR Dataset Generation
## Method 1 (Soumendra's)

## Method 2 (Aditya's)
The entire setup is broken into two steps-
1. Generate intermediate file for an employee
2. Create 1000 heatmaps for the same employee

### Step 1 (Intermediate Step)
```python
import gc
import time
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd  # type: ignore
from pydantic import BaseModel


class FTR(BaseModel):
    file_path: Union[str, Path]
    data: Optional[pd.DataFrame]
    num_hours: int = 120
    new_columns: List[str] = [f"worked_hr_{x}" for x in range(1, num_hours + 1)]
    columns: List[str] = [f"hour_{x}" for x in range(1, num_hours + 1)]

    class Config:
        arbitrary_types_allowed = True

    def process(self):
        self.load_data()
        self.process_interim()
        all_employees = self.data["EmpNo_Anon"].tolist()[:1]
        for emp in all_employees:
            subset = self.data.loc[self.data["EmpNo_Anon"] == emp]
            subset = self.process_single_employee(subset)
            subset.to_csv(f"data/interim/final_data_{emp}.csv")
            del subset
            gc.collect()

    def load_data(self) -> pd.DataFrame:
        if Path(self.file_path).exists():
            df = pd.read_csv(str(self.file_path), dtype={"EmpNo_Anon": int})
        else:
            raise ValueError(f"FTR data at path {self.file_path} doesn't exist!")

        # considering only employees which have faced an incident
        emps = df.loc[df["incident"] == True]["EmpNo_Anon"].tolist()
        df = df.loc[df["EmpNo_Anon"].isin(emps)]
        df = df[["EmpNo_Anon", "Work_DateTime", "incident"]]
        df["Work_DateTime"] = pd.to_datetime(df["Work_DateTime"])
        df = df.sort_values(by=["EmpNo_Anon", "Work_DateTime"])

        self.data = df
        return self

    def add_hours_data(self):
        for n in range(1, self.num_hours + 1):
            self.data[f"hour_{n}"] = self.data["hour_diff_cumsum"] - self.data[
                "hour_diff_cumsum"
            ].shift(n)
        return self

    def process_interim(self):
        # calculate hour difference between consecutive records
        self.data["hour_diff"] = self.data["Work_DateTime"] - self.data[
            "Work_DateTime"
        ].shift(1)
        self.data["hour_diff"] = (
            self.data["hour_diff"].dt.days * 24
            + self.data["hour_diff"].dt.seconds // 3600
        )

        # removing first & last occurrence of employees as they cause issues
        heads = self.data.groupby("EmpNo_Anon").head(1).index.tolist()
        tails = self.data.groupby("EmpNo_Anon").tail(1).index.tolist()
        self.data = self.data.drop(axis=0, index=heads + tails)

        # calculate break between hours till date ever since an employee started working
        self.data["hour_diff_cumsum"] = self.data.groupby("EmpNo_Anon")[
            "hour_diff"
        ].cumsum()
        self.data["hour_diff_cumsum"] = self.data["hour_diff_cumsum"].fillna(0)

        self.add_hours_data()
        return self

    def process_single_employee(self, subset: pd.DataFrame) -> pd.DataFrame:
        for col in self.new_columns:
            subset.loc[:, col] = False

        subset["final"] = subset[self.columns].values.tolist()

        for idx, row in tqdm(subset.iterrows()):
            final_list = [int(x) for x in row["final"] if x < self.num_hours]
            cols = [f"worked_hr_{val}" for val in final_list]
            subset.loc[idx, cols] = True

        # drop non-required columns to reduce RAM usage
        subset = subset.drop(columns=self.columns)
        subset = subset.drop(columns=["hour_diff", "hour_diff_cumsum", "final"])
        return subset


if __name__ == "__main__":
    start = time.time()
    num_hours = 120
    ftr = FTR(file_path="data/raw/public.csv", num_hours=num_hours)
    ftr.process()
    print(f"FTR Step 1 (Execution Time): {time.time()-start} seconds")
```

```bash
RUN 1 - FTR Step 1 (Execution Time): 125.3 seconds
RUN 2 - FTR Step 1 (Execution Time): 113.0 seconds
RUN 3 - FTR Step 1 (Execution Time): 108.3 seconds
RUN 4 - FTR Step 1 (Execution Time): 134.0 seconds
RUN 5 - FTR Step 1 (Execution Time): 129.3 seconds

AVERAGE RUNTIME - 121.9 seconds
```

### Step 2 (HeatMap Generation)
```python
import time
from pathlib import Path
from typing import List, Optional, Union

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore
tqdm.pandas()

from .heatmap import Heatmap


def hour_to_heatmap(row, num_hours=120):
    worker = row["EmpNo_Anon"]
    label = 1 if row["incident"] == True else 0
    timestamp = row["Work_DateTime"]
    data = row[[f"worked_hr_{x}" for x in range(1, num_hours + 1)]].astype(float)
    data = data.values.reshape((num_hours // 24, 24))
    return worker, label, timestamp, data


def extract_one_worker_hour(file_path: Union[str, Path]):
    if Path(file_path).exists():
        df = pd.read_csv(str(file_path))
        df["Work_DateTime"] = pd.to_datetime(df["Work_DateTime"]).dt.strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
    else:
        raise ValueError(f"Data at path {file_path} doesn't exist!")
    diag = df.sample(1)
    worker, label, timestamp, data = hour_to_heatmap(diag)
    return worker, label, timestamp, data, diag


def generate_one_binary_mask(row):
    # Setup
    cmap = mpl.colors.ListedColormap(["w", "g"])
    bounds = [0.0, 0.5, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xticks(range(0, 24))
    plt.yticks(range(0, 5))
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    worker, label, timestamp, data = hour_to_heatmap(row, 120)

    ax.imshow(data, interpolation="none", cmap=cmap, norm=norm)
    fig.savefig(Path("data/processed") / f"{str(label)}_{str(worker)}_{timestamp}.png")
    plt.close(fig)


def generate_all_binary_masks(
    input_path: str, output_path: str = "data/processed", num_hours: int = 120
):
    if Path(input_path).exists():
        df = pd.read_csv(str(input_path)).head(1000)
        df["Work_DateTime"] = pd.to_datetime(df["Work_DateTime"]).dt.strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
    else:
        raise ValueError(f"Data at path {input_path} doesn't exist!")

    df.progress_apply(generate_one_binary_mask, axis=1)


if __name__ == "__main__":
    num_hours = 120

    start = time.time()
    generate_all_binary_masks(
        input_path="data/interim/final_data_38321907.csv", num_hours=num_hours
    )
    print(f"Heatmap Generation Runtime: {time.time()-start} seconds")
```

#### Output
```bash
## RUN 1
100%|█████████████████████████████████████████████████████████| 1000/1000 [03:58<00:00,  4.20it/s]
Heatmap Generation Runtime: 238.6475989818573 seconds

## RUN 2
100%|█████████████████████████████████████████████████████████| 1000/1000 [03:57<00:00,  4.21it/s]
Heatmap Generation Runtime: 237.70236206054688 seconds

## RUN 3
100%|█████████████████████████████████████████████████████████| 1000/1000 [03:31<00:00,  4.73it/s]
Heatmap Generation Runtime: 211.82737278938293 seconds

## RUN 4
100%|█████████████████████████████████████████████████████████| 1000/1000 [03:35<00:00,  4.65it/s]
Heatmap Generation Runtime: 215.28080534934998 seconds

## RUN 5
100%|█████████████████████████████████████████████████████████| 1000/1000 [03:34<00:00,  4.66it/s]
Heatmap Generation Runtime: 214.96475911140442 seconds
```

**AVERAGE RUNTIME**
```bash
223.64 seconds
```