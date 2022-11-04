from __future__ import annotations

import os
from typing import TYPE_CHECKING

from yacs.config import CfgNode as ConfigurationNode

from core.common.logging import get_logger

if TYPE_CHECKING:
    from typing import Optional


_logger = get_logger(__file__)

# TODO Avoid using `os.environ.get` and rather utilize `dotenv` module
_dataset_dir = os.environ.get("DATASET_DIR", "")

__C = ConfigurationNode()

__C.DATA = ConfigurationNode()
__C.DATA.IMAGE_SIDE_SIZE = 224
__C.DATA.VAL_PORTION = 0.15
__C.DATA.TEST_PORTION = 0.15
__C.DATA.PURE_BG_IMAGES_PORTION = 0.01
__C.DATA.FG_IMAGE_REP_COUNT = 200
__C.DATA.BG_LABEL = "background"
__C.DATA.IMAGE_FEATURE_NAME = "image"
__C.DATA.LABEL_FEATURE_NAME = "label"
__C.DATA.BBOX_FEATURE_NAME = "bbox"
__C.DATA.RAW_IMAGES_DATASET_DIR_PATH = os.path.join(_dataset_dir, "pandemic_image_dataset")
__C.DATA.TRAIN_DATASET_FILE_PATH = os.path.join(_dataset_dir, "pandemic_dataset_train.tfrecord")
__C.DATA.VAL_DATASET_FILE_PATH = os.path.join(_dataset_dir, "pandemic_dataset_val.tfrecord")
__C.DATA.TEST_DATASET_FILE_PATH = os.path.join(_dataset_dir, "pandemic_dataset_test.tfrecord")
__C.DATA.LABEL_INDEX_MAP_FILE_PATH = os.path.join(_dataset_dir, "label_index_map.json")
__C.DATA.RAND_SEED = 42

__C.MODEL = ConfigurationNode()
__C.MODEL.INPUT_SHAPE = (__C.DATA.IMAGE_SIDE_SIZE, __C.DATA.IMAGE_SIDE_SIZE, 3)
__C.MODEL.N_CLASSES = 161  # 160 types of card + 1 pure background
__C.MODEL.INPUTS_NAME = "image"
__C.MODEL.CLASSIFICATION_OUTPUTS_NAME = "class"
__C.MODEL.BBOX_OUTPUTS_NAME = "bbox"

__C.OPTIMIZER = ConfigurationNode()
__C.OPTIMIZER.NAME = "adam"
__C.OPTIMIZER.LEARNING_RATE = 1e-3
__C.OPTIMIZER.CLASSIFICATION_LOSS_WEIGHT = 0.5
__C.OPTIMIZER.BBOX_LOSS_WEIGHT = 0.5

__C.TRAIN = ConfigurationNode()
__C.TRAIN.N_EPOCHS = 30
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.VAL_FREQUENCY = 1
__C.TRAIN.PATIENCE = 3
__C.TRAIN.MONITOR_METRIC = "val_loss"
__C.TRAIN.N_WORKERS = 6
__C.TRAIN.REMOVE_PREV_CHECKPOINT = True
__C.TRAIN.CHECKPOINT_FILE_PATH = "./checkpoints/checkpoint.ckpt"
__C.TRAIN.TENSORBOARD_DIR_PATH = "./tensorboard_logs"
__C.TRAIN.RAND_SEED = 42

__C.EVAL = ConfigurationNode()
__C.EVAL.BATCH_SIZE = 16
__C.EVAL.N_WORKERS = 6


def get_config_defaults() -> ConfigurationNode:
    """Get a yacs `CfgNode` object with default values."""
    # Return a clone so that the defaults will not be altered
    return __C.clone()


def get_config_defaults_or_override(
    config_file_path: Optional[str] = None,
) -> ConfigurationNode:
    config = get_config_defaults()

    if config_file_path and os.path.exists(config_file_path):
        config.merge_from_file(config_file_path)
    else:
        _logger.warning(f"configuration file {config_file_path} specified but it does not exist")

    return config
