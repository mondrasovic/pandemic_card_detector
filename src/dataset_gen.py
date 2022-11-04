from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import argparse

    from typing import Dict, List

    from core.data.io import ClassificationDatasetWriter

from core.common.app import Application
from core.data.augment import (
    make_bg_image_augmenter,
    make_fg_bg_merged_image_augmenter,
    make_fg_image_augmenter,
)
from core.data.dataset import (
    SubsetType,
    find_image_file_paths,
    load_image_classification_dataset,
    make_fg_bg_dataset_generator,
)
from core.data.image import AugmentedForegroundImageImputer
from core.data.io import (
    LabelIndexMap,
    RawImageClassificationDatasetWriter,
    TFRecordClassificationDatasetWriter,
)
from core.data.serialization import make_example_serializer


class DatasetGeneratorApp(Application):
    def add_custom_parser_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "dataset_dir_path",
            type=str,
            help="directory path to the input dataset containing foreground classes",
        )
        parser.add_argument(
            "bg_images_dir_path",
            type=str,
            help="directory path to the input dataset containing background images",
        )

    def main(self) -> None:
        random.seed(self.config.DATA.RAND_SEED)
        np.random.seed(self.config.DATA.RAND_SEED)

        fg_images_dataset = load_image_classification_dataset(self.args.dataset_dir_path)
        bg_images_file_paths = find_image_file_paths(self.args.bg_images_dir_path)
        random.shuffle(bg_images_file_paths)

        label_index_map = LabelIndexMap(
            labels=itertools.chain((self.config.DATA.BG_LABEL,), fg_images_dataset.keys()),
            start_index=0,
        )

        fg_image_augmenter = make_fg_image_augmenter()
        bg_image_augmenter = make_bg_image_augmenter(self.config)
        merged_image_augmenter = make_fg_bg_merged_image_augmenter()

        fg_image_imputer = AugmentedForegroundImageImputer(
            fg_image_augmenter, bg_image_augmenter, merged_image_augmenter
        )

        dataset_writers: List[ClassificationDatasetWriter] = []

        if images_output_dir_path := self.config.DATA.RAW_IMAGES_DATASET_DIR_PATH:
            raw_images_dataset_writer = RawImageClassificationDatasetWriter(images_output_dir_path)
            dataset_writers.append(raw_images_dataset_writer)

        subset_to_file_path: Dict[SubsetType, str] = {}

        if train_dataset_file_path := self.config.DATA.TRAIN_DATASET_FILE_PATH:
            subset_to_file_path[SubsetType.TRAIN] = train_dataset_file_path
        if val_dataset_file_path := self.config.DATA.VAL_DATASET_FILE_PATH:
            subset_to_file_path[SubsetType.VAL] = val_dataset_file_path
        if test_dataset_file_path := self.config.DATA.TEST_DATASET_FILE_PATH:
            subset_to_file_path[SubsetType.TEST] = test_dataset_file_path

        if subset_to_file_path:
            example_serializer = make_example_serializer(self.config)
            tfrecord_dataset_writer = TFRecordClassificationDatasetWriter(
                subset_to_file_path, example_serializer, label_index_map
            )
            dataset_writers.append(tfrecord_dataset_writer)

        if not dataset_writers:
            self.log_warning("no dataset writer specified, exiting...")
            return

        dataset_gen = make_fg_bg_dataset_generator(
            fg_images_dataset, bg_images_file_paths, fg_image_imputer, self.config
        )
        dataset_gen.generate_dataset(*dataset_writers)  # type: ignore[arg-type]

        if label_index_map_file_path := self.config.DATA.LABEL_INDEX_MAP_FILE_PATH:
            self.log_info(f"generating label index map in file {label_index_map_file_path}")
            label_index_map.save_to_json(label_index_map_file_path)


if __name__ == "__main__":
    DatasetGeneratorApp(
        logger_name=__file__,
        description=(
            "Generates a synthetic dataset that contains foreground images "
            "on top of randomly selected backgrounds."
        ),
    )()
