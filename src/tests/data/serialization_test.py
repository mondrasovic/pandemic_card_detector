import pytest
import tensorflow as tf

from core.data.serialization import ExampleSerializer, make_example_serializer
from tests.utils import ColorType, build_image


@pytest.mark.usefixtures("fix_rand_seed")
class TestExampleSerializer:
    IMAGE_FEATURE_NAME = "image"
    LABEL_FEATURE_NAME = "label"
    BBOX_FEATURE_NAME = "bbox"

    @pytest.fixture
    def example_serializer(self):
        return ExampleSerializer(
            self.IMAGE_FEATURE_NAME, self.LABEL_FEATURE_NAME, self.BBOX_FEATURE_NAME
        )

    @pytest.fixture
    def image(self):
        return build_image(ColorType.RGB, 16, 16, to_tensor=True)

    @pytest.fixture
    def label(self):
        return 0

    @pytest.fixture
    def bbox(self):
        return tf.random.uniform(shape=(4,), dtype=tf.float32)

    def test_serialize_deserialize(self, example_serializer, image, label, bbox):
        example_proto = example_serializer.serialize(image, label, bbox)
        deserialized_data = example_serializer.deserialize(example_proto)

        assert isinstance(deserialized_data, dict)
        assert deserialized_data.keys() == {
            self.IMAGE_FEATURE_NAME,
            self.LABEL_FEATURE_NAME,
            self.BBOX_FEATURE_NAME,
        }

        assert tf.experimental.numpy.allclose(deserialized_data[self.IMAGE_FEATURE_NAME], image)
        assert deserialized_data[self.LABEL_FEATURE_NAME] == label
        assert tf.experimental.numpy.allclose(deserialized_data[self.BBOX_FEATURE_NAME], bbox)


class TestMakeFromConfig:
    def test_make_from_config(self, default_config):
        example_serializer = make_example_serializer(default_config)

        assert example_serializer.image_feature_name == default_config.DATA.IMAGE_FEATURE_NAME
        assert example_serializer.label_feature_name == default_config.DATA.LABEL_FEATURE_NAME
        assert example_serializer.bbox_feature_name == default_config.DATA.BBOX_FEATURE_NAME
