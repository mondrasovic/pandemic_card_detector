from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from typing import ByteString, Dict

    from yacs.config import CfgNode as ConfigurationNode


class ExampleSerializer:
    def __init__(
        self, image_feature_name: str, label_feature_name: str, bbox_feature_name: str
    ) -> None:
        self.image_feature_name = image_feature_name
        self.label_feature_name = label_feature_name
        self.bbox_feature_name = bbox_feature_name

        self.feature_description = {
            self.image_feature_name: tf.io.FixedLenFeature([], dtype=tf.string),
            self.label_feature_name: tf.io.FixedLenFeature([], dtype=tf.int64),
            self.bbox_feature_name: tf.io.FixedLenSequenceFeature(
                (), dtype=tf.float32, allow_missing=True
            ),
        }

    def serialize(self, image: tf.Tensor, label: int, bbox: tf.Tensor) -> ByteString:
        image_string = tf.io.encode_png(image)

        features = {
            self.image_feature_name: tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_string.numpy()])
            ),
            self.label_feature_name: tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            self.bbox_feature_name: tf.train.Feature(float_list=tf.train.FloatList(value=bbox)),
        }

        example = tf.train.Example(features=tf.train.Features(feature=features))
        example_proto = example.SerializeToString()

        return example_proto

    def deserialize(self, example_proto: ByteString) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(example_proto, self.feature_description)

        image = tf.io.decode_png(example[self.image_feature_name], channels=3)
        label = tf.cast(example[self.label_feature_name], tf.int32)
        bbox = tf.cast(tf.convert_to_tensor(example[self.bbox_feature_name]), tf.float32)

        return {
            self.image_feature_name: image,
            self.label_feature_name: label,
            self.bbox_feature_name: bbox,
        }


def make_example_serializer(config: ConfigurationNode) -> ExampleSerializer:
    return ExampleSerializer(
        image_feature_name=config.DATA.IMAGE_FEATURE_NAME,
        label_feature_name=config.DATA.LABEL_FEATURE_NAME,
        bbox_feature_name=config.DATA.BBOX_FEATURE_NAME,
    )
