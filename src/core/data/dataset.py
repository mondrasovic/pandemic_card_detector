from __future__ import annotations

import collections
import contextlib
import enum
import functools
import itertools
import math
import pathlib
import random
from typing import TYPE_CHECKING

import numpy as np
import tqdm

from core.data.image import read_image
from core.data.io import ClassificationDatasetWriter

if TYPE_CHECKING:
    from typing import Iterator, Mapping, MutableSequence, Sequence, Tuple

    from yacs.config import CfgNode as ConfigurationNode

    from core.data.image import AugmentedForegroundImageImputer, ForegroundImageImputer

    FilePathsList = MutableSequence[str]
    ImagesDataset = Mapping[str, FilePathsList]


class SubsetType(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SubsetTypeRandomGen:
    def __init__(self, dataset_size: int, val_portion: float = 0, test_portion: float = 0) -> None:
        self._check_in_range(val_portion, "validation dataset portion")
        self._check_in_range(test_portion, "test dataset portion")

        self.train_portion = 1 - (val_portion + test_portion)
        self._check_in_range(self.train_portion, "train dataset portion")
        self.val_portion = val_portion
        self.test_portion = test_portion

        self.dataset_size = dataset_size

        if abs((self.train_portion + self.val_portion + self.test_portion) - 1.0) > 1e-8:
            raise ValueError(f"train, val, and test dataset portions do not add up to 1")

    def __iter__(self) -> Iterator[SubsetType]:
        n_train_samples = int(math.ceil(self.dataset_size * self.train_portion))
        n_val_samples = int(math.floor(self.dataset_size * self.val_portion))
        n_test_samples = self.dataset_size - (n_train_samples + n_val_samples)

        subsets_seq = (
            [SubsetType.TRAIN] * n_train_samples
            + [SubsetType.VAL] * n_val_samples
            + [SubsetType.TEST] * n_test_samples
        )
        random.shuffle(subsets_seq)

        yield from iter(subsets_seq)

    @staticmethod
    def _check_in_range(val: float, label: str) -> None:
        if not (0 <= val <= 1):
            raise ValueError(f"{label} not in <0, 1> interval, got value {val}")


class ForegroundBackgroundDatasetGenerator:
    def __init__(
        self,
        fg_images_dataset: ImagesDataset,
        bg_images_file_path: FilePathsList,
        fg_image_imputer: ForegroundImageImputer,
        fg_image_rep_count: int = 1,
        pure_bg_images_portion: float = 0.5,
        val_portion: float = 0.2,
        test_portion: float = 0.2,
        bg_label: str = "background",
    ) -> None:
        self.fg_images_dataset = fg_images_dataset
        self.bg_images_file_path = bg_images_file_path
        self.fg_image_imputer = fg_image_imputer
        self.fg_image_rep_count = fg_image_rep_count
        self.pure_bg_images_portion = pure_bg_images_portion
        self.val_portion = val_portion
        self.test_portion = test_portion
        self.bg_label = bg_label

    def generate_dataset(self, *dataset_writers: Tuple[ClassificationDatasetWriter, ...]) -> None:
        with contextlib.ExitStack() as stack:
            dataset_writers_context: Sequence[ClassificationDatasetWriter] = [
                stack.enter_context(dataset_writer) for dataset_writer in dataset_writers  # type: ignore[arg-type]
            ]

            dataset_size = self._calc_output_dataset_size()
            subset_type_iter = iter(
                SubsetTypeRandomGen(dataset_size, self.val_portion, self.test_portion)
            )

            with tqdm.tqdm(total=dataset_size, desc="generating dataset") as pbar:
                for image, label, bbox in self._generate_images():
                    subset = next(subset_type_iter)

                    for dataset_writer in dataset_writers_context:
                        dataset_writer.add_example(subset, image, label, bbox)

                    pbar.update(1)

    def _generate_images(self) -> Iterator[Tuple[np.ndarray, str, np.ndarray]]:
        label_image_file_path_pairs = list(
            itertools.chain.from_iterable(
                [label_image_file_path_pair] * self.fg_image_rep_count
                for label_image_file_path_pair in self._iter_foreground_images_dataset()
            )
        )

        generated_fg_images_count = len(label_image_file_path_pairs)
        n_pure_bg_images = self._calc_n_pure_bg_images(generated_fg_images_count)

        label_image_file_path_pairs.extend((self.bg_label, "") for _ in range(n_pure_bg_images))

        random.shuffle(label_image_file_path_pairs)

        bg_images_file_path_iter = itertools.cycle(self.bg_images_file_path)

        read_image_cached = functools.lru_cache(read_image)

        for label, image_file_path in label_image_file_path_pairs:
            fg_image = read_image_cached(image_file_path) if image_file_path else None

            bg_image_file_path = next(bg_images_file_path_iter)
            bg_image = read_image(bg_image_file_path)

            merged_image, bbox = self.fg_image_imputer(bg_image, fg_image)

            if bbox is None:  # Pure background images do not have BBOX of the foreground object
                # yxyx format
                bbox = np.asarray(
                    (0, 0, merged_image.shape[1] - 1, merged_image.shape[0] - 1), np.float32
                )

            yield merged_image, label, bbox

    def _calc_output_dataset_size(self) -> int:
        fg_base_images_count = sum(
            len(images_file_path) for images_file_path in self.fg_images_dataset.values()
        )
        generated_fg_images_count = fg_base_images_count * self.fg_image_rep_count
        n_pure_bg_images = self._calc_n_pure_bg_images(generated_fg_images_count)
        final_size = generated_fg_images_count + n_pure_bg_images

        return final_size

    def _calc_n_pure_bg_images(self, generated_fg_images_count: int) -> int:
        return int(round(generated_fg_images_count * self.pure_bg_images_portion))

    def _iter_foreground_images_dataset(self) -> Iterator[Tuple[str, str]]:
        for label, images_file_path in self.fg_images_dataset.items():
            for image_file_path in images_file_path:
                yield (label, image_file_path)


def load_image_classification_dataset(dataset_dir_path: str) -> ImagesDataset:
    dataset_dir = pathlib.Path(dataset_dir_path)
    dataset = collections.defaultdict(list)

    for image_file in dataset_dir.glob("*/*.*"):
        parent_label = image_file.parent.stem
        child_label = image_file.stem
        label = parent_label + "-" + child_label
        dataset[label].append(str(image_file))

    dataset.default_factory = None

    return dataset


def find_image_file_paths(root_dir_path: str, recurse: bool = True) -> FilePathsList:
    root_dir = pathlib.Path(root_dir_path)
    pattern = "*.jpg"

    if recurse:
        files_iter = root_dir.rglob(pattern)
    else:
        files_iter = root_dir.glob(pattern)

    image_file_paths = [str(image_file) for image_file in files_iter]

    return image_file_paths


def make_fg_bg_dataset_generator(
    fg_images_dataset: ImagesDataset,
    bg_images_file_paths: FilePathsList,
    fg_image_imputer: AugmentedForegroundImageImputer,
    config: ConfigurationNode,
) -> ForegroundBackgroundDatasetGenerator:
    return ForegroundBackgroundDatasetGenerator(
        fg_images_dataset,
        bg_images_file_paths,
        fg_image_imputer,
        config.DATA.FG_IMAGE_REP_COUNT,
        config.DATA.PURE_BG_IMAGES_PORTION,
        config.DATA.VAL_PORTION,
        config.DATA.TEST_PORTION,
        config.DATA.BG_LABEL,
    )
