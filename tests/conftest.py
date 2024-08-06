import random

import pytest


@pytest.fixture
def fixed_random_seed() -> None:
    random.seed(1447)
