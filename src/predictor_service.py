from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flask import Flask, g, jsonify, request
from PIL import Image

from core.common.app import Application
from core.data.io import make_label_index_map
from core.modeling.model import make_predictor_from_checkpoint

if TYPE_CHECKING:
    from flask import Response

app = Flask(__name__)

config_ = None
predictor_ = None
label_index_map_ = None


@app.route("/")
@app.route("/index")
def index() -> Response:
    # TODO Print help here.
    return "Pandemic image predictor"


@app.route("/predict-image", methods=["POST"])
def predict_image() -> Response:
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["image"]

    image_orig = Image.open(file)
    image = np.asarray(image_orig)

    prediction = predictor_.predict_images(image)

    label_index = prediction[config_.MODEL.CLASSIFICATION_OUTPUTS_NAME][0]
    assert label_index.ndim == 0
    label = label_index_map_.get_index_label(int(label_index))
    print(f"predicted class: {label}")

    bbox_prediction = prediction[config_.MODEL.BBOX_OUTPUTS_NAME][0]
    assert len(bbox_prediction) == 4 and bbox_prediction.ndim == 1
    bbox_xyxy = (
        int(bbox_prediction[1]),
        int(bbox_prediction[0]),
        int(bbox_prediction[3]),
        int(bbox_prediction[2]),
    )

    prediction_response = {"label": label, "bbox_xyxy": bbox_xyxy}

    return jsonify(prediction_response)


class RESTFulPredictorApp(Application):
    # TODO Add options to specify port and host for the REST application
    def main(self) -> None:
        global config_
        global predictor_
        global label_index_map_

        config_ = self.config.clone()
        predictor_ = make_predictor_from_checkpoint(self.config)
        label_index_map_ = make_label_index_map(self.config)

        app.run(debug=True, host="0.0.0.0")


if __name__ == "__main__":
    cli_app = RESTFulPredictorApp(
        __file__,
        "Predictor that localizes and classifies Pandemic cards in a given image using REST API.",
    )()
