from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Tuple

    from core.data.augment import AugTransformer


class ForegroundImageImputer(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, bg_image: np.ndarray, fg_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass


class AugmentedForegroundImageImputer(ForegroundImageImputer):
    def __init__(
        self,
        fg_image_augmenter: Optional[AugTransformer] = None,
        bg_image_augmenter: Optional[AugTransformer] = None,
        merged_image_augmenter: Optional[AugTransformer] = None,
    ) -> None:
        self.fg_image_augmenter = fg_image_augmenter
        self.bg_image_augmenter = bg_image_augmenter
        self.merged_image_augmenter = merged_image_augmenter

    def __call__(
        self, bg_image: np.ndarray, fg_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        bg_image_rgba = bgr_to_rgba(bg_image)
        augmented_bg_image = self._augment_if_possible(self.bg_image_augmenter, bg_image_rgba)
        assert augmented_bg_image is not None

        if fg_image is None:
            augmented_fg_image = None
        else:
            fg_image_rgba = bgr_to_rgba(fg_image)  # type: ignore[arg-type]
            augmented_fg_image = self._augment_if_possible(self.fg_image_augmenter, fg_image_rgba)

        if augmented_fg_image is None:
            bbox = None
        else:
            bbox = self._insert_foreground_to_background_image(
                augmented_fg_image, augmented_bg_image
            )

        merged_image = self._augment_if_possible(self.merged_image_augmenter, augmented_bg_image)
        merged_image_bgr = rgba_to_bgr(merged_image)  # type: ignore[arg-type]

        return merged_image_bgr, bbox

    @staticmethod
    def _insert_foreground_to_background_image(
        fg_image: np.ndarray,
        bg_image: np.ndarray,
        fg_bg_max_min_side_ratio: float = 0.8,
        interpolation: int = cv.INTER_LANCZOS4,
    ) -> np.ndarray:
        if not (0.0 < fg_bg_max_min_side_ratio <= 1.0):
            raise ValueError(
                "foregound to background min side ratio outside of (0, 1> interval, "
                f"got {fg_bg_max_min_side_ratio}"
            )

        if fg_image.shape[-1] != 4:
            raise ValueError(
                f"expected foreground image to be RGBA, got {fg_image.shape[-1]} channels"
            )

        if not (3 <= bg_image.shape[-1] <= 4):
            raise ValueError(
                f"expected background image to be RGB or RGBA, got {bg_image.shape[-1]} channels"
            )

        # TODO Make this parametric
        fg_bg_max_min_side_ratio = np.random.random() * (0.9 - 0.6) + 0.6

        bg_height, bg_width = bg_image.shape[:2]
        fg_max_side = min(bg_height, bg_width) * fg_bg_max_min_side_ratio
        scale = fg_max_side / max(fg_image.shape[:2])

        scaled_fg_image = cv.resize(
            fg_image, dsize=None, fx=scale, fy=scale, interpolation=interpolation
        )

        scaled_fg_height, scaled_fg_width = scaled_fg_image.shape[:2]
        max_x_shift = bg_width - scaled_fg_width
        if max_x_shift == 0:
            x_shift = 0
        else:
            x_shift = np.random.randint(low=0, high=bg_width - scaled_fg_width)

        max_y_shift = bg_height - scaled_fg_height
        if max_y_shift == 0:
            y_shift = 0
        else:
            y_shift = np.random.randint(low=0, high=bg_height - scaled_fg_height)

        x1 = x_shift
        y1 = y_shift
        x2 = x_shift + scaled_fg_width
        y2 = y_shift + scaled_fg_height
        bbox = np.asarray((y1, x1, y2, x2))

        mask = np.expand_dims(scaled_fg_image[:, :, 3] > 0, axis=-1)  # Add alpha channel.
        np.copyto(bg_image[y1:y2, x1:x2, :3], scaled_fg_image[:, :, :-1], where=mask)

        return bbox

    @staticmethod
    def _augment_if_possible(
        augmenter: Optional[AugTransformer], image: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if (augmenter is None) or (image is None):
            return image
        return augmenter(image)


def read_image(image_file_path: str) -> np.ndarray:
    return cv.imread(image_file_path, cv.IMREAD_COLOR)


def bgr_to_rgba(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, cv.COLOR_BGR2RGBA)


def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, cv.COLOR_RGBA2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
