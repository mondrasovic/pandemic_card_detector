import random

import numpy as np
import pytest

from core.data.dataset import SubsetType
from core.data.io import (
    LabelIndexMap,
    TFRecordClassificationDatasetWriter,
    read_tfrecord_dataset_from_config,
)
from core.data.serialization import make_example_serializer
from tests.utils import ColorType, build_image


class TestLabelIndexMap:
    @pytest.fixture
    def tmp_json_file(self, tmpdir):
        return tmpdir.join("label_index_map.json")

    @pytest.fixture
    def tmp_json_file_path(self, tmp_json_file):
        return str(tmp_json_file)

    @pytest.fixture
    def label_index_map(self):
        return LabelIndexMap(("a", "b", "c"), start_index=1)

    def test_len(self, label_index_map):
        assert len(label_index_map) == 3

    def test_getitem(self, label_index_map):
        assert label_index_map["a"] == 1
        assert label_index_map["b"] == 2
        assert label_index_map["c"] == 3

    def test_labels_preserve_order(self, label_index_map):
        assert label_index_map.labels == ("a", "b", "c")

    def test_get_index_label(self, label_index_map):
        assert label_index_map.get_index_label(1) == "a"
        assert label_index_map.get_index_label(2) == "b"
        assert label_index_map.get_index_label(3) == "c"

    def test_save_to_json_and_from_json(self, label_index_map, tmp_json_file_path):
        label_index_map.save_to_json(tmp_json_file_path)
        loaded_label_index_map = LabelIndexMap.from_json(tmp_json_file_path)

        assert label_index_map.label_to_index == loaded_label_index_map.label_to_index
        assert label_index_map.index_to_label == loaded_label_index_map.index_to_label

    @pytest.mark.parametrize(
        "json_content",
        [
            pytest.param('{"a": 0, "b": 1, "c": 0}', id="duplicate_indices"),
            pytest.param('{"a": -1, "b": 0, "c": 1}', id="negative_index"),
            pytest.param('{"a": 0, "b": 1, "c": 3}', id="nonconsecutive_indices"),
        ],
    )
    def test_load_from_json_raises_when_invalid_indices(self, json_content, tmp_json_file):
        tmp_json_file.write(json_content)
        with pytest.raises(ValueError):
            tmp_json_file_path = str(tmp_json_file)
            LabelIndexMap.from_json(tmp_json_file_path)

    def test_raises_when_empty(self):
        with pytest.raises(ValueError):
            LabelIndexMap(())


@pytest.mark.usefixtures("fix_rand_seed")
class TestTFRecordDatasetReading:
    N_SAMPLES = 16
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64

    @pytest.fixture
    def label_index_map(self):
        return LabelIndexMap(("class_1", "class_2", "class_3"), start_index=0)

    @pytest.fixture
    def data_examples(self, label_index_map):
        examples = []

        for _ in range(self.N_SAMPLES):
            image = build_image(ColorType.GRAYSCALE, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
            label = random.choice(label_index_map.labels)
            bbox = np.random.uniform(size=4)

            examples.append((SubsetType.TRAIN, image, label, bbox))

        return examples

    @pytest.fixture
    def example_serializer(self, default_config):
        return make_example_serializer(default_config)

    @pytest.fixture
    def tfrecord_dataset_file_path(
        self, tmpdir, example_serializer, label_index_map, data_examples
    ):
        dataset_file_path = str(tmpdir.join("tmp_tfrecord_dataset.tfrecord"))
        subset_to_file_path = {SubsetType.TRAIN: dataset_file_path}

        with TFRecordClassificationDatasetWriter(
            subset_to_file_path, example_serializer, label_index_map
        ) as writer:
            for data_example in data_examples:
                writer.add_example(*data_example)

        return dataset_file_path

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_read_tfrecord_dataset(
        self, tfrecord_dataset_file_path, example_serializer, batch_size, default_config
    ):
        dataset = read_tfrecord_dataset_from_config(
            default_config,
            tfrecord_dataset_file_path,
            example_serializer,
            batch_size=batch_size,
            shuffle_buffer_size_coef=8,
        )
        single_batch = next(iter(dataset))

        assert isinstance(single_batch, tuple)
        assert len(single_batch) == 2

        inputs = single_batch[0]
        assert isinstance(inputs, dict)
        assert inputs.keys() == {default_config.MODEL.INPUTS_NAME}

        outputs_true = single_batch[1]
        assert isinstance(outputs_true, dict)
        assert outputs_true.keys() == {
            default_config.MODEL.CLASSIFICATION_OUTPUTS_NAME,
            default_config.MODEL.BBOX_OUTPUTS_NAME,
        }

        self._test_samples_shape(
            inputs,
            default_config.MODEL.INPUTS_NAME,
            (batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3),
        )
        self._test_samples_shape(
            outputs_true, default_config.MODEL.CLASSIFICATION_OUTPUTS_NAME, (batch_size,)
        )
        self._test_samples_shape(
            outputs_true, default_config.MODEL.BBOX_OUTPUTS_NAME, (batch_size, 4)
        )

    @classmethod
    def _test_samples_shape(cls, batch_data, feature_name, expected_shape):
        assert batch_data[feature_name].shape == expected_shape
