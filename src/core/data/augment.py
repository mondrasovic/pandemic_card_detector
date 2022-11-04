from __future__ import annotations

from typing import TYPE_CHECKING

import albumentations as A
import cv2 as cv
import numpy as np

if TYPE_CHECKING:
    from albumentations.core.composition import TransformType
    from yacs.config import CfgNode as ConfigurationNode


class AugTransformer:
    def __init__(self, transform: TransformType) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray, *, force_apply: bool = False) -> np.ndarray:
        return self.transform(image=image, force_apply=force_apply)["image"]


def make_fg_image_augmenter(p: float = 1.0) -> AugTransformer:
    return AugTransformer(
        transform=A.Sequential(
            [
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.8),
                A.OneOf(
                    transforms=[
                        A.Blur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.GaussianBlur(p=0.3),
                        A.MotionBlur(p=0.2),
                        A.ZoomBlur(max_factor=1.05, p=0.2),
                    ],
                    p=1.0,
                ),
                A.Perspective(
                    scale=0.1,
                    p=0.5,
                    fit_output=True,
                    pad_val=(0, 0, 0, 0),
                    interpolation=cv.INTER_LANCZOS4,
                ),
            ],
            p=p,
        )
    )


def make_bg_image_augmenter(config: ConfigurationNode, p: float = 1.0) -> AugTransformer:
    image_side_size = config.DATA.IMAGE_SIDE_SIZE
    return AugTransformer(
        transform=A.Sequential(
            [
                A.HorizontalFlip(p=0.2),
                A.SmallestMaxSize(
                    max_size=image_side_size,
                    interpolation=cv.INTER_LANCZOS4,
                    always_apply=True,
                ),
                A.RandomCrop(height=image_side_size, width=image_side_size, always_apply=True),
            ],
            p=p,
        )
    )


def make_fg_bg_merged_image_augmenter(p: float = 0.5) -> AugTransformer:
    return AugTransformer(
        transform=A.Sequential(
            [
                A.GaussNoise(p=0.3),
                A.OpticalDistortion(border_mode=cv.BORDER_REFLECT_101, p=0.1),
            ],
            p=p,
        )
    )
