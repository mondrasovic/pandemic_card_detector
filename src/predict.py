from __future__ import annotations

from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

from core.common.app import Application
from core.data.io import make_label_index_map
from core.modeling.model import make_predictor_from_checkpoint

if TYPE_CHECKING:
    import argparse


class PredictorApp(Application):
    def add_custom_parser_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input_image_file_path",
            type=str,
            help="input image file path to execute prediction on",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output_image_file_path",
            type=str,
            default=None,
            help="output image file path to visualize the prediction",
        )

    def main(self) -> None:
        image_orig = cv.imread(self.args.input_image_file_path, cv.IMREAD_COLOR)
        image_rgb = cv.cvtColor(image_orig, cv.COLOR_BGR2RGB)

        self.log_info(
            f"running prediction on image {self.args.input_image_file_path} "
            f"of shape {image_orig.shape}"
        )
        predictor = make_predictor_from_checkpoint(self.config)
        prediction = predictor.predict_images(image_rgb)

        label_index_map = make_label_index_map(self.config)
        label_index = prediction[self.config.MODEL.CLASSIFICATION_OUTPUTS_NAME][0]
        assert label_index.ndim == 0
        label = label_index_map.get_index_label(int(label_index))
        print(f"predicted class: {label}")

        bbox_prediction = prediction[self.config.MODEL.BBOX_OUTPUTS_NAME][0]
        assert len(bbox_prediction) == 4 and bbox_prediction.ndim == 1
        print(f"predicted bounding box: {bbox_prediction}")

        image_visualized = image_orig.copy()
        draw_image_bbox(image_visualized, bbox_prediction)

        if self.args.output_image_file_path:
            cv.imwrite(self.args.output_image_file_path, image_visualized)


def draw_image_bbox(image: np.ndarray, bbox_yxyx: np.ndarray) -> None:
    bbox_yxyx = bbox_yxyx.round().astype(np.int32)
    cv.rectangle(
        image,
        (bbox_yxyx[1], bbox_yxyx[0]),
        (bbox_yxyx[3], bbox_yxyx[2]),
        color=(0, 255, 0),
        thickness=2,
        lineType=cv.LINE_AA,
    )


if __name__ == "__main__":
    PredictorApp(
        logger_name=__file__,
        description="Predictor that localizes and classifies Pandemic cards in a given image.",
    )()
