from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel


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
