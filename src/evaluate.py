from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import classification_report

from core.common.app import Application
from core.data.io import make_label_index_map, read_tfrecord_dataset_from_config
from core.data.serialization import make_example_serializer
from core.modeling.model import make_predictor_from_checkpoint

if TYPE_CHECKING:
    from typing import Sequence


class EvaluatorApp(Application):
    def main(self) -> None:
        test_dataset_file_path = self.config.DATA.TEST_DATASET_FILE_PATH
        batch_size = self.config.EVAL.BATCH_SIZE

        example_serializer = make_example_serializer(self.config)
        predictor = make_predictor_from_checkpoint(self.config)
        label_index_map = make_label_index_map(self.config)

        self.log_info(f"reading test dataset for evaluation from file {test_dataset_file_path}")
        test_dataset = read_tfrecord_dataset_from_config(
            self.config, test_dataset_file_path, example_serializer, batch_size
        )

        self.log_info(f"starting evaluation with batch size of {batch_size} samples")

        classification_outputs_name = self.config.MODEL.CLASSIFICATION_OUTPUTS_NAME
        targets_batches, predictions_batches = [], []
        for inputs, targets in test_dataset:
            predictions = predictor(inputs)

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
