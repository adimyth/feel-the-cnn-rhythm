from typing import Optional, Union

import matplotlib.pyplot as plt  # type:ignore
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import seaborn as sns  # type:ignore
from pydantic import BaseModel


class Heatmap(BaseModel):
    data: Optional[Union[pd.DataFrame, np.ndarray]]

    class Config:
        arbitrary_types_allowed = True

    def create_heatmap(self):
        fig, _ = plt.subplots(figsize=(10, 5))
        _ = sns.heatmap(self.data, linewidths=1, cmap="Greens", linecolor="white", cbar=False)
        plt.axis("off")
        return fig
