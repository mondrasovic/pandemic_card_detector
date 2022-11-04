from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

from core.common.app import Application
from core.modeling.model import make_compiled_model

if TYPE_CHECKING:
    import argparse


class ShowModelApp(Application):
    def add_custom_parser_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-o",
            "--output",
            dest="output_image_file_path",
            type=str,
            default="model_architecture.png",
            help="output image file path to store the model architecture",
        )

    def main(self) -> None:
        object_localizer_classifier = make_compiled_model(self.config)
        tf.keras.utils.plot_model(
            object_localizer_classifier.model,
            to_file=self.args.output_image_file_path,
            show_shapes=True,
            show_layer_activations=True,
            show_layer_names=True,
            dpi=128,
        )

        print(object_localizer_classifier.model.summary())


if __name__ == "__main__":
    ShowModelApp(
        logger_name=__file__, description="Shows model architecture in both diagram and text form."
    )()
