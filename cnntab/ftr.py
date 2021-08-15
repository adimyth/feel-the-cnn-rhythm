import gc
from datetime import date, timedelta
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
import random


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
        return self


if __name__ == "__main__":
    x = (
        FTR(file_path="data/public.csv")
        .load_data()
        .get_employee_record(sample=False)
        .employee_records.to_csv("data/final.csv")
    )
