import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btlab import data  # noqa: E402


@pytest.fixture(scope="session")
def noise():
    return data.random_walks(n_days=1500, n_assets=6, seed=3)
