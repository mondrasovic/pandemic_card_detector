import functools

import numpy as np
import pytest
import tensorflow as tf

from core.modeling.model import ObjectLocalizerAndClassifier
from tests.utils import ColorType, build_images

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
N_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)
N_CLASSES = 10
INPUTS_NAME = "inputs"
CLASSIFICATION_OUTPUTS_NAME = "classification"
BBOX_OUTPUTS_NAME = "bbox"

build_test_images = functools.partial(
    build_images, color_type=ColorType.RGB, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
)


class _FakeLoss(tf.keras.losses.Loss):
    def __call__(self, y_true, y_pred, sample_weight=None) -> tf.Tensor:
        return tf.constant(0.0)


class TestObjectLocalizerAndClassifier:
    @pytest.fixture
    def model(self):
        assert INPUT_SHAPE[0] == INPUT_SHAPE[1]

        model = ObjectLocalizerAndClassifier(
            INPUT_SHAPE,
            N_CLASSES,
            INPUTS_NAME,
            CLASSIFICATION_OUTPUTS_NAME,
            BBOX_OUTPUTS_NAME,
        )

        loss = _FakeLoss()
        loss_weights = {}
        optimizer = tf.keras.optimizers.SGD()
        metrics = {
            model.classification_outputs_name: tf.keras.metrics.SparseCategoricalAccuracy(),
            model.bbox_outputs_name: tf.keras.metrics.MeanAbsoluteError(),
        }

        model.compile(loss, loss_weights, optimizer, metrics)

        return model

    @pytest.mark.parametrize(
        "images",
        [
            pytest.param(build_test_images(count=1), id="single_image"),
            pytest.param(build_test_images(count=2), id="two_images"),
            pytest.param(build_test_images(count=4), id="four_images"),
        ],
    )
    @pytest.mark.usefixtures("fix_rand_seed", "tf_clear_session")
    def test_predict_images(self, model, images):
        outputs = model.predict_images(images)

        assert isinstance(outputs, dict)
        assert outputs.keys() == {
            CLASSIFICATION_OUTPUTS_NAME,
            BBOX_OUTPUTS_NAME,
        }

        n_images = len(images) if images.ndim > 3 else 1

        classification_outputs = outputs[CLASSIFICATION_OUTPUTS_NAME]
        assert isinstance(classification_outputs, np.ndarray)
        assert classification_outputs.shape == (n_images,)
        assert tf.reduce_all(tf.greater_equal(classification_outputs, 0))
        assert tf.reduce_all(tf.less(classification_outputs, N_CLASSES))

        bbox_outputs = outputs[BBOX_OUTPUTS_NAME]
        assert isinstance(bbox_outputs, np.ndarray)
        assert bbox_outputs.shape == (n_images, 4)
