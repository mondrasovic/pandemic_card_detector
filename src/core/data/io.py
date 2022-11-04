from __future__ import annotations

import abc
import itertools
import json
import operator
import pathlib
import shutil
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
import tensorflow as tf

from core.data.image import bgr_to_rgb

if TYPE_CHECKING:
    from typing import Any, ByteString, Dict, Iterable, Sequence, Tuple

    from yacs.config import CfgNode as ConfigurationNode

    from core.data.dataset import SubsetType
    from core.data.serialization import ExampleSerializer

    FilePathsList = Sequence[str]
    ImagesDataset = Dict[str, FilePathsList]


class ClassificationDatasetWriter(abc.ABC):
    def __enter__(self) -> ClassificationDatasetWriter:
        self.start()

        return self

    def __exit__(self, ext_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()

    @abc.abstractmethod
    def add_example(
        self, subset: SubsetType, image: np.ndarray, label: str, bbox: np.ndarray
    ) -> None:
        pass

    def start(self) -> None:
        pass

    def finish(self) -> None:
        pass


class RawImageClassificationDatasetWriter(ClassificationDatasetWriter):
    def __init__(
        self,
        root_dir_path: str,
        file_ext: str = "jpg",
        start_id: int = 0,
        clear_directory: bool = True,
    ) -> None:
        self.root_dir = pathlib.Path(root_dir_path)
        self.file_ext = file_ext
        self.image_id_iter = itertools.count(start=start_id)
        self.clear_directory = clear_directory

    def start(self) -> None:
        if self.clear_directory:
            shutil.rmtree(str(self.root_dir), ignore_errors=True)

    def add_example(
        self, subset: SubsetType, image: np.ndarray, label: str, bbox: np.ndarray
    ) -> None:
        label_dir = self.root_dir / subset.value / label
        label_dir.mkdir(parents=True, exist_ok=True)

        image_id = next(self.image_id_iter)
        image_file_name = f"{image_id:05d}.{self.file_ext}"
        image_file_path = str(label_dir / image_file_name)

        image = image.copy()
        self._draw_image_bbox(image, bbox)

        cv.imwrite(image_file_path, image)

    @staticmethod
    def _draw_image_bbox(image: np.ndarray, bbox_yxyx: np.ndarray) -> None:
        bbox_yxyx = bbox_yxyx.round().astype(np.int32)
        cv.rectangle(
            image,
            (bbox_yxyx[1], bbox_yxyx[0]),
            (bbox_yxyx[3], bbox_yxyx[2]),
            color=(0, 255, 0),
            thickness=2,
            lineType=cv.LINE_AA,
        )


class TFRecordClassificationDatasetWriter(ClassificationDatasetWriter):
    def __init__(
        self,
        subset_to_file_path: Dict[SubsetType, str],
        example_serializer: ExampleSerializer,
        label_index_map: LabelIndexMap,
    ) -> None:
        self.subset_to_file_path = subset_to_file_path
        self.example_serializer = example_serializer
        self.label_index_map = label_index_map
        self.subset_to_writer: Dict[SubsetType, tf.io.TFRecordWriter] = {}

    def start(self) -> None:
        for subset_type, output_file_path in self.subset_to_file_path.items():
            writer = tf.io.TFRecordWriter(output_file_path)
            self.subset_to_writer[subset_type] = writer

    def finish(self) -> None:
        for writer in self.subset_to_writer.values():
            writer.close()

    def add_example(
        self, subset: SubsetType, image: np.ndarray, label: str, bbox: np.ndarray
    ) -> None:
        writer = self.subset_to_writer[subset]
        self._add_example(writer, image, label, bbox)

    def _add_example(
        self,
        writer: tf.io.TFRecordWriter,
        image: np.ndarray,
        label: str,
        bbox: np.ndarray,
    ) -> None:
        image_rgb = bgr_to_rgb(image)
        image_tensor = tf.convert_to_tensor(image_rgb, tf.uint8)
        label_index = self.label_index_map[label]
        bbox_tensor = tf.convert_to_tensor(bbox, tf.float32)

        example_proto = self.example_serializer.serialize(image_tensor, label_index, bbox_tensor)
        writer.write(example_proto)


class LabelIndexMap:
    def __init__(self, labels: Iterable[str], start_index: int = 0) -> None:
        self.label_to_index = {}
        self.index_to_label = {}

        for index, label in enumerate(labels, start=start_index):
            self.label_to_index[label] = index
            self.index_to_label[index] = label

        if not self.label_to_index:
            raise ValueError("label index map cannot be empty")

    def __len__(self) -> int:
        return len(self.label_to_index)

    def __getitem__(self, label: str) -> int:
        return self.label_to_index[label]

    @property
    def labels(self) -> Sequence[str]:
        return tuple(self.label_to_index.keys())

    def get_index_label(self, index: int) -> str:
        return self.index_to_label[index]

    def save_to_json(self, json_file_path: str) -> None:
        with open(json_file_path, "wt") as out_file:
            json.dump(
                self.label_to_index,
                out_file,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )

    @staticmethod
    def from_json(json_file_path: str) -> LabelIndexMap:
        with open(json_file_path, "rt") as in_file:
            label_index_map_dict = json.load(in_file)

        n_labels = len(label_index_map_dict)
        unique_indices = set(label_index_map_dict.values())
        if n_labels != len(unique_indices):
            raise ValueError("label indices have to be unique")

        label_index_map_sorted = sorted(label_index_map_dict.items(), key=operator.itemgetter(1))
        min_index = label_index_map_sorted[0][1]
        if min_index < 0:
            raise ValueError("label indices must be non-negative")

        max_index = label_index_map_sorted[-1][1]
        if (max_index - min_index + 1) != n_labels:
            raise ValueError("label indices need to be consecutive")

        return LabelIndexMap((label for label, _ in label_index_map_sorted), start_index=min_index)


def read_tfrecord_dataset(
    dataset_file_path: str,
    example_serializer: ExampleSerializer,
    inputs_name: str,
    classification_outputs_name: str,
    bbox_outputs_name: str,
    batch_size: int = 16,
    *,
    shuffle_buffer_size_coef: int = 256,
) -> tf.data.Dataset:
    @tf.function
    def map_example(
        example_proto: ByteString,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        example = example_serializer.deserialize(example_proto)
        inputs = {inputs_name: example[example_serializer.image_feature_name]}
        outputs_true = {
            classification_outputs_name: example[example_serializer.label_feature_name],
            bbox_outputs_name: example[example_serializer.bbox_feature_name],
        }
        return inputs, outputs_true

    dataset = tf.data.TFRecordDataset(dataset_file_path)
    dataset = dataset.map(map_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer_size_coef > 0:
        dataset = dataset.shuffle(buffer_size=batch_size * shuffle_buffer_size_coef)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def read_tfrecord_dataset_from_config(
    config: ConfigurationNode,
    dataset_file_path: str,
    example_serializer: ExampleSerializer,
    batch_size: int = 16,
    *,
    shuffle_buffer_size_coef: int = 256,
) -> tf.data.Dataset:
    return read_tfrecord_dataset(
        dataset_file_path,
        example_serializer,
        inputs_name=config.MODEL.INPUTS_NAME,
        classification_outputs_name=config.MODEL.CLASSIFICATION_OUTPUTS_NAME,
        bbox_outputs_name=config.MODEL.BBOX_OUTPUTS_NAME,
        batch_size=batch_size,
        shuffle_buffer_size_coef=shuffle_buffer_size_coef,
    )
