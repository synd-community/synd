import pandas as pd


class ObesityDataset:
    def __init__(self, path: str = "data/obesity.csv"):
        data = pd.read_csv(path)

        self.categorical_columns = [
            "Gender",
            "family_history_with_overweight",
            "FAVC",
            "CAEC",
            "SMOKE",
            "SCC",
            "CALC",
            "MTRANS",
            "Obesity_level",
        ]
        for col in self.categorical_columns:
            data[col] = data[col].astype("category")

        self.data = data
