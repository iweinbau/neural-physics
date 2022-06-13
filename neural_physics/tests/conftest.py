import numpy as np
import pytest


@pytest.fixture
def dummy_data() -> np.ndarray:
    num_vertices = 1000
    num_frames = 100
    X = np.random.rand(3 * num_vertices, num_frames)

    # dim_external = 16
    # Y = np.random.rand(dim_external, num_frames)

    return X
