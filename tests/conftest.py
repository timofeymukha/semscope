import os

import numpy as np
import pytest

DATA_DIR = "/tmp/timofey/code/pySEMTools/examples/data"
MIXLAY = os.path.join(DATA_DIR, "mixlay0.f00001")
CYLINDER = os.path.join(DATA_DIR, "cylinder_mesh0.f00000")
FLAT_PLATE_META = "/tmp/sa-flat-average-smoke.wMUkFV/flat_plate_2d0.nek5000"


def need(path):
    if not os.path.exists(path):
        pytest.skip(f"sample data {path} not available")
    return path


@pytest.fixture
def mixlay():
    return need(MIXLAY)


@pytest.fixture
def cylinder():
    return need(CYLINDER)


@pytest.fixture
def flat_plate_meta():
    return need(FLAT_PLATE_META)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
