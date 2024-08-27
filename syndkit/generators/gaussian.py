import warnings

warnings.filterwarnings("ignore")
import random
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from syndkit.base import DataGenerator


class Preprocessor:
    def __init__(self, data):
        self.int_features = data.select_dtypes(include=["int64"]).columns.tolist()
        self.float_features = data.select_dtypes(include=["float64"]).columns.tolist()
        self.categorical_columns = data.select_dtypes(
            include=["category"]
        ).columns.tolist()

        self.categorical_encoders = dict()
        self.one_hot_encoders = dict()
        self.encoded_vars = dict()
        self.scaler = StandardScaler()
        self.columns = data.columns

        # iterate over all categorical columns and fit one-hot encoders
        for column in self.categorical_columns:
            self.one_hot_encoders[column] = OneHotEncoder().fit(
                np.asarray(data[column].astype("category")).reshape(-1, 1)
            )
            self.encoded_vars[column] = (
                self.one_hot_encoders[column]
                .transform(np.asarray(data[column].astype("category")).reshape(-1, 1))
                .toarray()
            )

    def encode(self, data):
        for column in self.categorical_columns:
            data = data.drop([column], axis=1)

            # compute the inverse sigmoid function for each one-hot encoded column
            for i in range(0, self.encoded_vars[column].shape[1]):
                data[column + str(i)] = self.encoded_vars[column][:, i]
                data[column + str(i)] = data[column + str(i)].astype("int8")
                data[column + str(i)] = np.exp(
                    np.asarray(data[column + str(i)].values)
                ) / (1 + np.exp(np.asarray(data[column + str(i)].values)))

        # standardize the data
        scaled_data = self.scaler.fit_transform(data.values)

        # return the preprocessed data
        return pd.DataFrame(scaled_data, columns=data.columns.tolist())

    def decode(self, generated_samples):
        # inverse standardization of data
        generated_samples[generated_samples.columns] = self.scaler.inverse_transform(
            generated_samples[generated_samples.columns]
        )

        # convert the integer attributes of the data frame
        for c in self.int_features:
            generated_samples[c] = generated_samples[c].astype("int64")
            synthetic_data = generated_samples.select_dtypes(include=["int64"])

        # convert the float attributes of the data frame
        for c in self.float_features:
            generated_samples[c] = generated_samples[c].astype("float64")
            synthetic_data = generated_samples.select_dtypes(include=["float64"])

        # transform categorical features to original features types
        for col in self.categorical_columns:

            # get the obtained numerical values of each categorical attribute encoded group
            cols_drop = (generated_samples.filter(regex=col)).columns.tolist()
            values = generated_samples.filter(regex=col).values
            generated_samples = generated_samples.drop(cols_drop, axis=1)

            # iterate over all values of assign a 1 to the maximum value row, to the rest it gives a value of 0
            for i in range(0, values.shape[0]):
                m = max(values[i, :])
                for j, k in enumerate(values[i, :]):
                    if k == m:
                        values[i, j] = 1
                    else:
                        values[i, j] = 0

            # perform the inverse one-hot encoding of the categorical attribute
            generated_samples[col] = self.one_hot_encoders[col].inverse_transform(
                values
            )

        # sort the attributes of the dataframe to be in the same order as in real train data
        synthetic_data = pd.DataFrame(columns=self.columns)
        for col in self.columns:
            synthetic_data[col] = generated_samples[col]

        # return the transformed synthetic dataframe
        return synthetic_data


class GaussianMultivariateGenerator(DataGenerator):
    def __init__(self, model: GaussianMultivariate, preprocessor: Preprocessor):
        self._model = model
        self.preprocessor = preprocessor

    @classmethod
    def train(cls, dataset: Sequence[Any]):
        preprocessor = Preprocessor(dataset)
        encoded = preprocessor.encode(dataset)
        gm = GaussianMultivariate()
        gm.fit(encoded)
        return cls(gm, preprocessor)

    def generate(self, samples: int, **kwargs) -> Sequence[Any]:
        return self._model.sample(samples)
