from typing import Any, List, Optional, Self, Sequence, Tuple

from syndkit.base import DataGenerator


class StrawmanGenerator(DataGenerator):
    def __init__(self, samples: List[Tuple[Any, Any]]):
        self._samples = samples

    @classmethod
    def train(cls, dataset: Sequence[Any], samples: Optional[int] = None) -> Self:
        return cls(dataset[:samples])

    def generate(self, samples: int, **kwargs) -> Sequence[Any]:
        return self._samples.sample(samples, replace=True)
