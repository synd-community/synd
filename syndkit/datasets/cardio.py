import pandas as pd
from ctgan import CTGAN


def load(
    path: str = "data/B_Cardio_Data_Real.csv",
    columns: list[str] = [
        "gender",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "cardio",
    ],
):
    data = pd.read_csv(path, delimiter=";")

    ctgan = CTGAN()
    ctgan.fit(data, columns)

    synthetic_data = ctgan.sample(len(data))
    synthetic_data
