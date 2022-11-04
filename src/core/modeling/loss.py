from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf
import tensorflow_addons as tfa

if TYPE_CHECKING:
    from typing import Dict

    from yacs.config import CfgNode as ConfigurationNode


def make_loss(config: ConfigurationNode) -> Dict[str, tf.keras.losses.Loss]:
    return {
        config.MODEL.CLASSIFICATION_OUTPUTS_NAME: tf.keras.losses.SparseCategoricalCrossentropy(
            name="cross-entropy"
        ),
        config.MODEL.BBOX_OUTPUTS_NAME: tf.keras.losses.MeanSquaredError(name="MSE"),
        config.MODEL.BBOX_OUTPUTS_NAME: tfa.losses.GIoULoss(mode="iou", name="IoU"),
    }


def make_loss_weights(config: ConfigurationNode) -> Dict[str, float]:
    return {
        config.MODEL.CLASSIFICATION_OUTPUTS_NAME: config.OPTIMIZER.CLASSIFICATION_LOSS_WEIGHT,
        config.MODEL.BBOX_OUTPUTS_NAME: config.OPTIMIZER.BBOX_LOSS_WEIGHT,
    }
