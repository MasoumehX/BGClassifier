import os
import pandas as pd


def read_csv_file(path, separator=","):
    return pd.read_csv(path, sep=separator)
