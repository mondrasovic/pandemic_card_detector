import numpy as np
import pytest

from core.data.augment import (
    make_bg_image_augmenter,
    make_fg_bg_merged_image_augmenter,
    make_fg_image_augmenter,
)
from tests.utils import ColorType, build_image


@pytest.mark.usefixtures("fix_rand_seed")
class TestImageAugmenters:
    @pytest.fixture
    def image_rgba(self, default_config):
        return build_image(
            ColorType.RGBA, default_config.DATA.IMAGE_SIDE_SIZE, default_config.DATA.IMAGE_SIDE_SIZE
        )

    @pytest.fixture
    def image_rgb(self, default_config):
        return build_image(
            ColorType.RGB, default_config.DATA.IMAGE_SIDE_SIZE, default_config.DATA.IMAGE_SIDE_SIZE
        )

    @pytest.fixture
    def fg_image_augmenter(self):
        return make_fg_image_augmenter()

    @pytest.fixture
    def bg_image_augmenter(self, default_config):
        return make_bg_image_augmenter(default_config)

    @pytest.fixture
    def fg_bg_merged_image_augmenter(self):
        return make_fg_bg_merged_image_augmenter()

    @pytest.mark.parametrize("image_fixture", ["image_rgba"])
    def test_fg_image_augmenter(self, fg_image_augmenter, image_fixture, request):
        self._test_augmentation(fg_image_augmenter, image_fixture, request)

    @pytest.mark.parametrize("image_fixture", ["image_rgb", "image_rgba"])
    def test_bg_image_augmenter(self, bg_image_augmenter, image_fixture, request):
        self._test_augmentation(bg_image_augmenter, image_fixture, request)

    @pytest.mark.parametrize("image_fixture", ["image_rgb", "image_rgba"])
    def test_fg_bg_merged_image_augmenter(
        self, fg_bg_merged_image_augmenter, image_fixture, request
    ):
        self._test_augmentation(fg_bg_merged_image_augmenter, image_fixture, request)

    def _test_augmentation(self, augmenter, input_image_fixture, request):
        input_image = request.getfixturevalue(input_image_fixture)
        augmented_image = augmenter(input_image, force_apply=True)

        assert augmented_image.dtype == input_image.dtype
        assert augmented_image.shape == input_image.shape
        assert not np.allclose(augmented_image, input_image)
