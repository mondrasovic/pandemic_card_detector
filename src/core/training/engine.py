from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from typing import Any, List, Optional

    from yacs.config import CfgNode as ConfigurationNode


# TODO Find out what the type of the "history object" returned from the `fit` method actually is
def do_train(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    n_epochs: int,
    callbacks: Optional[List[tf.keras.callbacks.Callback]],
    val_dataset: Optional[tf.data.Dataset] = None,
    val_frequency: int = 1,
    n_workers: int = 1,
) -> Any:
    validate = val_dataset is not None

    params = {"epochs": n_epochs, "callbacks": callbacks}
    if n_workers > 1:
        params["use_multiprocessing"] = True
        params["workers"] = n_workers
    if validate:
        params["validation_data"] = val_dataset
        params["validation_freq"] = val_frequency

    history = model.fit(train_dataset, **params)

    return history


def do_train_from_config(
    config: ConfigurationNode,
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    val_dataset: Optional[tf.data.Dataset] = None,
) -> Any:
    return do_train(
        model,
        train_dataset,
        n_epochs=config.TRAIN.N_EPOCHS,
        callbacks=callbacks,
        val_dataset=val_dataset,
        val_frequency=config.TRAIN.VAL_FREQUENCY,
        n_workers=config.TRAIN.N_WORKERS,
    )
