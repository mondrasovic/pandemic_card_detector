from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from typing import Dict

    from yacs.config import CfgNode as ConfigurationNode


def make_metrics(config: ConfigurationNode) -> Dict[str, tf.keras.metrics.Metric]:
    return {
        config.MODEL.CLASSIFICATION_OUTPUTS_NAME: tf.keras.metrics.SparseCategoricalAccuracy(
            name="acc"
        ),
        config.MODEL.BBOX_OUTPUTS_NAME: tf.keras.metrics.MeanAbsoluteError(name="MAE"),
    }
