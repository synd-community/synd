""" Workflow sketch

from typing import Any, Callable, Sequence, Type

from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from syndkit.base import DataGenerator

class ClassificationUtilityEvaluation:
    def __init__(
        self,
        dataset: Sequence[Any],
        generator_type: Type[DataGenerator],
        classifier: Pipeline,
        compute_metrics: Callable = precision_recall_fscore_support,
    ):
        train, test = dataset.split()

        # build a synthetic generation model from the dataset
        generator = generator_type.train(train)
        synthetic_train = generator.sample(len(dataset))

        # fit models on the real and synthetic data
        model = classifier.fit(train)
        synthetic_model = classifier.fit(synthetic_train)

        # predict with real and synthetic derived models
        predictions = model.predict(test)
        synthetic_predictions = synthetic_model.predict(test)

        # compute and compare their predictions in terms of test set metrics
        metrics = compute_metrics(test, predictions)
        synthetic_metrics = compute_metrics(test, synthetic_predictions)
        return synthetic_metrics - metrics
"""
