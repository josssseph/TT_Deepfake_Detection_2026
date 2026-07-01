import math

import pytest

from deepfake_app.errors import InferenceError
from deepfake_app.inference import label_from_score


def test_threshold_is_inclusive() -> None:
    assert label_from_score(0.3329, 0.333) == 0
    assert label_from_score(0.3330, 0.333) == 1
    assert label_from_score(0.9000, 0.333) == 1


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_score_is_rejected(value: float) -> None:
    with pytest.raises(InferenceError):
        label_from_score(value, 0.333)
