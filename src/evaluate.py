from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import classification_report

from core.common.app import Application
from core.data.io import LabelIndexMap, read_tfrecord_dataset_from_config
from core.data.serialization import make_example_serializer
from core.modeling.model import make_compiled_model

if TYPE_CHECKING:
    from typing import Iterable, Sequence


class EvaluatorApp(Application):
    def main(self) -> None:
        test_dataset_file_path = self.config.DATA.TEST_DATASET_FILE_PATH
        batch_size = self.config.EVAL.BATCH_SIZE

        example_serializer = make_example_serializer(self.config)

        object_localizer_classifier = make_compiled_model(self.config)
        object_localizer_classifier.model.load_weights(
            self.config.TRAIN.CHECKPOINT_FILE_PATH
        ).expect_partial()

        label_index_map = LabelIndexMap.from_json(self.config.DATA.LABEL_INDEX_MAP_FILE_PATH)

        self.log_info(f"reading test dataset for evaluation from file {test_dataset_file_path}")
        test_dataset = read_tfrecord_dataset_from_config(
            self.config, test_dataset_file_path, example_serializer, batch_size
        )

        self.log_info(f"starting evaluation with batch size of {batch_size} samples")

        classification_outputs_name = self.config.MODEL.CLASSIFICATION_OUTPUTS_NAME
        targets_batches, predictions_batches = [], []
        model = object_localizer_classifier.model
        for inputs, targets in test_dataset:
            predictions = model(inputs)

            predictions_batches.append(predictions[classification_outputs_name].numpy())
            targets_batches.append(targets[classification_outputs_name].numpy())

        report = evaluate_classification(
            targets_batches, predictions_batches, label_index_map.labels
        )
        print(report)


def evaluate_classification(
    targets_batches: Sequence[np.ndarray],
    predictions_batches: Sequence[np.ndarray],
    labels: Sequence[str],
):
    all_targets = np.concatenate(targets_batches)
    all_label_indices = np.concatenate(
        [np.argmax(prediction, axis=1) for prediction in predictions_batches]
    )
    report = classification_report(all_targets, all_label_indices, target_names=labels)
    return report


if __name__ == "__main__":
    EvaluatorApp(
        logger_name=__file__,
        description="Runs model evaluation on the test dataset and prints a report.",
    )()
