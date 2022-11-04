from __future__ import annotations

import abc
import functools
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from typing import Dict, Tuple

    from yacs.config import CfgNode as ConfigurationNode

    InputShape = Tuple[int, int, int]


class PredictorMixin(abc.ABC):
    @abc.abstractmethod
    def predict_images(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        pass


class ObjectLocalizerAndClassifier(PredictorMixin):
    def __init__(
        self,
        input_shape: InputShape,
        n_classes: int,
        inputs_name: str = "inputs",
        classification_outputs_name: str = "class",
        bbox_outputs_name: str = "bbox",
    ) -> None:
        if input_shape[0] != input_shape[1]:
            raise ValueError(f"expected input shape of a square image, got {input_shape}")

        self.n_classes = n_classes
        self.inputs_name = inputs_name
        self.classification_outputs_name = classification_outputs_name
        self.bbox_outputs_name = bbox_outputs_name

        self.model = self._build_model(input_shape)

    # PredictorMixin
    def predict_images(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        elif images.ndim != 4:
            raise ValueError(f"images expected to be 3- or 4-dimensional, got {images.ndim} dims")

        inputs = {self.inputs_name: tf.convert_to_tensor(images, tf.float32)}
        predictions = self.model.predict(inputs)
        label_predictions = predictions[self.classification_outputs_name]
        bbox_predictions = predictions[self.bbox_outputs_name]

        label_indices = tf.argmax(label_predictions, axis=1).numpy()
        bboxes = tf.cast(bbox_predictions.round(), tf.int32).numpy()

        return {
            self.classification_outputs_name: label_indices,
            self.bbox_outputs_name: bboxes,
        }

    def compile_model(
        self,
        loss: Dict[str, tf.keras.losses.Loss],
        loss_weights: Dict[str, float],
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: Dict[str, tf.keras.metrics.Metric],
    ) -> None:
        self.model.compile(
            optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights
        )

    def _build_model(self, input_shape: InputShape) -> tf.keras.Model:
        input_layer = tf.keras.layers.Input(shape=input_shape, name=self.inputs_name)
        preprocessed_features = self._build_preprocessing(input_layer)
        backbone_features = self._build_backbone(preprocessed_features)
        classification_outputs = self._build_classification_head(backbone_features)
        bbox_outputs = self._build_bbox_head(backbone_features)

        inputs = {self.inputs_name: input_layer}
        outputs = {
            self.classification_outputs_name: classification_outputs,
            self.bbox_outputs_name: bbox_outputs,
        }

        return tf.keras.Model(inputs=inputs, outputs=outputs, name="object_loc_and_cls")

    @staticmethod
    def _build_preprocessing(input_features: tf.Tensor) -> tf.Tensor:
        return tf.keras.layers.Rescaling(scale=1.0 / 255, name="rescale")(input_features)

    @classmethod
    def _build_backbone(cls, input_features: tf.Tensor) -> tf.Tensor:
        x = build_3x3_conv_block("block_1", input_features, n_filters=64)
        x = build_3x3_conv_block("block_2", x, n_filters=128)
        x = tf.keras.layers.MaxPool2D(name="max_pool_1")(x)

        x = build_3x3_conv_block("block_3", x, n_filters=128)
        x = build_3x3_conv_block("block_4", x, n_filters=256)
        x = tf.keras.layers.MaxPool2D(name="max_pool_2")(x)

        return x

    def _build_classification_head(self, backbone_features: tf.Tensor) -> tf.Tensor:
        get_name_ = functools.partial(get_name, "cls")

        x = build_3x3_conv_block("cls", backbone_features, n_filters=256)
        x = tf.keras.layers.GlobalAveragePooling2D(name=get_name_("gap"))(x)
        x = tf.keras.layers.Dense(512, activation="relu", name=get_name_("fc"))(x)
        x = tf.keras.layers.Dropout(0.2, name=get_name_("dropout"))(x)
        x = tf.keras.layers.Dense(
            self.n_classes, activation="softmax", name=self.classification_outputs_name
        )(x)

        return x

    def _build_bbox_head(self, backbone_features: tf.Tensor) -> tf.Tensor:
        get_name_ = functools.partial(get_name, "bbox")

        x = build_3x3_conv_block("bbox", backbone_features, n_filters=512)
        x = tf.keras.layers.MaxPooling2D(name=get_name_("max_pool"))(x)
        x = build_1x1_conv("bbox", x, n_filters=32)
        x = tf.keras.layers.Flatten(name=get_name_("flat"))(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name=get_name_("fc"))(x)
        x = tf.keras.layers.Dense(units=4, activation="linear", name=self.bbox_outputs_name)(x)

        return x


def build_3x3_conv_block(
    name_prefix: str, prev_x: tf.Tensor, n_filters: int, use_batch_norm: bool = True
) -> tf.Tensor:
    get_name_ = functools.partial(get_name, name_prefix)

    x = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=3,
        strides=(1, 1),
        activation="relu",
        padding="same",
        use_bias=(not use_batch_norm),
        name=get_name_("conv_3x3"),
    )(prev_x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name=get_name_("bn"))(x)

    return x


def build_1x1_conv(name_prefix: str, prev_x: tf.Tensor, n_filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=1,
        strides=(1, 1),
        activation="relu",
        padding="same",
        use_bias=True,
        name=get_name(name_prefix, "conv_1x1"),
    )(prev_x)

    return x


def get_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def make_model(config: ConfigurationNode) -> ObjectLocalizerAndClassifier:
    model = ObjectLocalizerAndClassifier(
        input_shape=config.MODEL.INPUT_SHAPE,
        n_classes=config.MODEL.N_CLASSES,
        inputs_name=config.MODEL.INPUTS_NAME,
        classification_outputs_name=config.MODEL.CLASSIFICATION_OUTPUTS_NAME,
        bbox_outputs_name=config.MODEL.BBOX_OUTPUTS_NAME,
    )

    return model


def make_compiled_model(config: ConfigurationNode) -> ObjectLocalizerAndClassifier:
    from core.modeling.loss import make_loss, make_loss_weights
    from core.modeling.model import make_model
    from core.training.metrics import make_metrics
    from core.training.optimizer import make_optimizer

    model = make_model(config)
    optimizer = make_optimizer(config)
    loss = make_loss(config)
    loss_weights = make_loss_weights(config)
    metrics = make_metrics(config)

    model.compile_model(loss, loss_weights, optimizer, metrics)

    return model
