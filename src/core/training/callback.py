from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from typing import List

    from yacs.config import CfgNode as ConfigurationNode


def make_callbacks(config: ConfigurationNode) -> List[tf.keras.callbacks.Callback]:
    monitor_metric = config.TRAIN.MONITOR_METRIC

    callbacks = []

    if config.TRAIN.VAL_FREQUENCY > 0:  # Validation mode enabled
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=config.TRAIN.PATIENCE,
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)

    checkpoint_file_path = config.TRAIN.CHECKPOINT_FILE_PATH
    if checkpoint_file_path:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            monitor=monitor_metric,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        callbacks.append(model_checkpoint)

    tensorboard_dir_path = config.TRAIN.TENSORBOARD_DIR_PATH
    if tensorboard_dir_path:
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir_path)
        callbacks.append(tensorboard)

    return callbacks
