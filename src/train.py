import tensorflow as tf

from core.common.app import Application
from core.data.io import read_tfrecord_dataset_from_config
from core.data.serialization import make_example_serializer
from core.modeling.model import make_compiled_model
from core.training.callback import make_callbacks
from core.training.engine import do_train_from_config


class TrainerApp(Application):
    def main(self) -> None:
        tf.random.set_seed(self.config.TRAIN.RAND_SEED)

        train_dataset_file_path = self.config.DATA.TRAIN_DATASET_FILE_PATH
        val_dataset_file_path = self.config.DATA.VAL_DATASET_FILE_PATH
        batch_size = self.config.TRAIN.BATCH_SIZE

        example_serializer = make_example_serializer(self.config)

        self.log_info(f"reading training dataset from file {train_dataset_file_path}")
        train_dataset = read_tfrecord_dataset_from_config(
            self.config, train_dataset_file_path, example_serializer, batch_size
        )

        self.log_info(f"reading validation dataset from file {val_dataset_file_path}")
        val_dataset = read_tfrecord_dataset_from_config(
            self.config,
            val_dataset_file_path,
            example_serializer,
            batch_size,
            shuffle_buffer_size_coef=0,
        )

        object_localizer_classifier = make_compiled_model(self.config)
        callbacks = make_callbacks(self.config)

        self.log_info(
            f"training for {self.config.TRAIN.N_EPOCHS} with "
            f"batch size of {self.config.TRAIN.BATCH_SIZE}"
        )

        do_train_from_config(
            self.config, object_localizer_classifier.model, train_dataset, callbacks, val_dataset
        )


if __name__ == "__main__":
    TrainerApp(__file__, "Initiates model training with the given configuration.")()
