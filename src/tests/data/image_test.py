import pytest

from core.data.image import AugmentedForegroundImageImputer
from tests.utils import ColorType, build_image


class TestForegroundImageImputer:
    @pytest.mark.parametrize(
        "fg_image_color_type, bg_image_color_type, expected_message",
        [
            pytest.param(
                ColorType.GRAYSCALE,
                ColorType.RGB,
                "expected foreground image to be RGBA",
                id="fg_image_grayscale",
            ),
            pytest.param(
                ColorType.RGB,
                ColorType.RGB,
                "expected foreground image to be RGBA",
                id="fg_image_rgb",
            ),
            pytest.param(
                ColorType.RGBA,
                ColorType.GRAYSCALE,
                "expected background image to be RGB or RGBA",
                id="bg_image_grayscale",
            ),
        ],
    )
    def test_insert_foreground_to_background_image_raises_for_invalid_n_channels(
        self, fg_image_color_type, bg_image_color_type, expected_message
    ):
        with pytest.raises(ValueError) as exc_info:
            fg_image = build_image(fg_image_color_type, 32, 32)
            bg_image = build_image(bg_image_color_type, 64, 64)
            AugmentedForegroundImageImputer._insert_foreground_to_background_image(
                fg_image, bg_image
            )

            message = str(exc_info.value)
            assert expected_message in message
