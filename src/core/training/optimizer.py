from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from yacs.config import CfgNode as ConfigurationNode


def make_optimizer(config: ConfigurationNode) -> tf.keras.optimizers.Optimizer:
    optimizer_name = config.OPTIMIZER.NAME

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(config.OPTIMIZER.LEARNING_RATE)
    else:
        raise ValueError(f"unrecognized optimizer name, expected 'adam', got {optimizer_name}")
