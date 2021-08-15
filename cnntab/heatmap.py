from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from typing import Union, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Heatmap(BaseModel):
    data: Optional[Union[pd.DataFrame, np.ndarray]]

    class Config:
        arbitrary_types_allowed = True

    def create_heatmap(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1 = sns.heatmap(
            self.data, linewidths=1, cmap="Greens", linecolor="white", cbar=False
        )
        plt.axis("off")
        return fig
