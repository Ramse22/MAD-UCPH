import numpy as np
import pandas as pd


def read_data(file_name):
    data = pd.read_csv(file_name)
    data_norm = data
    for i in data:
        for j in data:
            data_norm[i][j] = (data[i][j]  - max(data[j]))/(min(data(j))  - max(data[j])