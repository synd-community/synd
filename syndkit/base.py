from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Self, Tuple, TypeVar

DatasetT = TypeVar("DatasetT", bound=List[Tuple[Any, Any]])
MetricsT = TypeVar("MetricsT", bound=Dict[str, float])
OutputT = TypeVar("OutputT")


class DataGenerator:
    def train(cls, dataset: DatasetT, **kwargs) -> Self: ...

    def generate(self, samples: int, **kwargs) -> List[Any]: ...


class Model(ABC):
    @classmethod
    @abstractmethod
    def train(cls, dataset: DatasetT, **kwargs) -> Self: ...

    @abstractmethod
    def predict(self, data: Any) -> OutputT: ...

    @classmethod
    def evaluate(cls, dataset: DatasetT) -> MetricsT:
        model = cls.train(dataset)
        inputs, labels = zip(*dataset)
        predictions = [model.predict(i) for i in inputs]
        return model.compute_metrics(predictions, labels)

    @classmethod
    def compute_metrics(
        cls, predictions: List[OutputT], labels: List[OutputT]
    ) -> MetricsT: ...

    @classmethod
    def synthetic_evaluate(
        cls, synth_dataset: DatasetT, real_dataset: DatasetT
    ) -> MetricsT:
        sy_metrics = cls.evaluate(synth_dataset)
        rl_metrics = cls.evaluate(real_dataset)
        return {k: sy_metrics[k] - rl_metrics[k] for k in sy_metrics.keys()}
