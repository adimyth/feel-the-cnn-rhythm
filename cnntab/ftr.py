from datetime import date, timedelta
import gc
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm


def get_diff(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df[f"hour{n}"] = df["hour_diff_cumsum"] - df["hour_diff_cumsum"].shift(n)
    return df


class FTR(BaseModel):
    file_path: str = ""
    base_path: str = ""
    new_columns: List[str] = [f"worked_hr_{x}" for x in range(1, 121)]
    columns: List[str] = [f"hour{x}" for x in range(1, 121)]

    def process(self):
        sanitised_data = self.load_data()
        interim_data = self.process_interim(sanitised_data)
        all_employees = interim_data["EmpNo_Anon"].tolist()
        for emp in all_employees:
            subset = interim_data.loc[interim_data["EmpNo_Anon"]==emp]
            subset = self.process_single_employee(subset)
            subset.to_csv(self.base_path+f"final_data_{emp}.csv")
            del subset
            gc.collect()

    def load_data(self) -> pd.DataFrame:
        """
        Loads raw data & sanitises it

        Returns:
            pd.DataFrame: Returns sanitised data
        """
        # load raw data
        df = pd.read_csv(self.file_path)

        # considering only employees which have faced an incident
        emps = df.loc[df["incident"] == True]["EmpNo_Anon"].tolist()
        df = df.loc[df["EmpNo_Anon"].isin(emps)]
        df = df[["EmpNo_Anon", "Work_DateTime", "incident"]]

        df["Work_DateTime"] = pd.to_datetime(df["Work_DateTime"], errors="coerce")
        # sort values by "Employee" & "Work_DateTime"
        df = df.sort_values(by=["EmpNo_Anon", "Work_DateTime"])
        return df

    def process_interim(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds record for whether someone had worked 1<x<120 hours ago

        Args:
            df (pd.DataFrame): Sanitised Data

        Returns:
            pd.DataFrame: Data with additional 120 columns
        """
        # calculate hour difference between consecutive records
        df["temp"] = df["Work_DateTime"].shift(1)
        df["hour_diff"] = df["Work_DateTime"] - df["temp"]
        df["hour_diff"] = (
            df["hour_diff"].dt.days * 24 + df["hour_diff"].dt.seconds // 3600
        )

        # removing first & last occurrence of employees as they cause issues
        heads = df.groupby("EmpNo_Anon").head(1).index.tolist()
        tails = df.groupby("EmpNo_Anon").tail(1).index.tolist()
        df = df.drop(axis=0, index=heads + tails)
        df = df.drop(columns=["temp"])

        # calculate break between hours till date ever since he/she started working
        df["hour_diff_cumsum"] = df.groupby("EmpNo_Anon")["hour_diff"].cumsum()
        df["hour_diff_cumsum"] = df["hour_diff_cumsum"].fillna(0)

        for n in tqdm(range(1, 121)):
            df = get_diff(df, n)
        return df


    def process_single_employee(self, subset: pd.DataFrame, emp_no: float) -> pd.DataFrame:
        for col in self.new_columns:
            subset[col] = False

        subset["final"] = subset[self.columns].values.tolist()

        for idx, row in tqdm(subset.iterrows()):
            final_list = [int(x) for x in row["final"] if x<120]
            cols = [f"worked_hr_{val}" for val in final_list]
            subset.loc[idx, cols] = True

        # drop non-required columns
        subset = subset.drop(columns=self.columns)
        subset = subset.drop(columns=["final"])
        return subset