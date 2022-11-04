import random

import numpy as np
import pytest
import tensorflow as tf

from core.common.config import get_config_defaults


@pytest.fixture
def fix_rand_seed():
    np_prev_state = np.random.get_state()
    random_prev_state = random.getstate()

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    yield

    np.random.set_state(np_prev_state)
    random.setstate(random_prev_state)


@pytest.fixture
def tf_clear_session():
    yield
    tf.keras.backend.clear_session()


@pytest.fixture
def default_config():
    config = get_config_defaults()
    config.freeze()
    return config
