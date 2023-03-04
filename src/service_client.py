from __future__ import annotations

import abc
import threading
import dataclasses
import queue
import pathlib
from typing import TYPE_CHECKING, Optional

# from core.common.app import Application

import requests
import numpy as np
import cv2 as cv

if TYPE_CHECKING:
    import argparse

    from typing import Any, Iterator, Tuple


@dataclasses.dataclass(frozen=True)
class ImageAnnotation:
    label: str
    bbox_xyxy: Tuple[int, int, int, int]


@dataclasses.dataclass
class ImageData:
    image_orig: np.ndarray
    image_preprocessed: Optional[np.ndarray] = None
    annotation : Optional[ImageAnnotation] = None


class ClosableQueue(queue.Queue):
    _SENTINEL = object()  # An indicator that no more elements are in the queue

    def close(self) -> None:
        self.put(self._SENTINEL)

    def __iter__(self) -> Iterator[Any]:
        while True:
            item = self.get()
            try:
                if item is self._SENTINEL:
                    return  # Cause the thread to exit
                yield item
            finally:
                self.task_done()


class StoppableWorker(threading.Thread):
    def __init__(self, func, in_queue, out_queue) -> None:
        super().__init__()

        self.func = func
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self) -> None:
        for item in self.in_queue:
            result = self.func(item)
            self.out_queue.put(result)


class ImageReader(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        pass


class DirectoryImageReader(ImageReader):
    def __init__(self, images_root_dir_path: str) -> None:
        super().__init__()

        self.images_root_dir_path = images_root_dir_path

    def __iter__(self) -> Iterator[np.ndarray]:
        for file in pathlib.Path(self.images_root_dir_path).iterdir():
            image = cv.imread(str(file), cv.IMREAD_COLOR)
            yield image


class CameraImageReader(ImageReader):
    def __iter__(self) -> Iterator[np.ndarray]:
        capture = cv.VideoCapture(0)

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            yield frame

        capture.close()


class ImagePreprocessor:
    def __init__(self, output_image_size: Tuple[int, int]) -> None:
        self.output_image_size = output_image_size

    def __call__(self, image_bgr: np.ndarray) -> Any:
        # image_cropped_bgr = square_crop(image_bgr)
        image_cropped_bgr = image_bgr
        image_rgb = cv.cvtColor(image_cropped_bgr, cv.COLOR_BGR2RGB)
        image_resized = cv.resize(
            image_rgb, self.output_image_size, interpolation=cv.INTER_LANCZOS4
        )
        return image_resized


class ImagePredictor:
    def __init__(self, prediction_url: str) -> None:
        self.prediction_service_url = prediction_url

    def __call__(self, image: np.ndarray) -> ImageAnnotation:
        _, encoded_image = cv.imencode(".png", image)
        response = requests.post(self.prediction_service_url, files={"image": encoded_image})
        if response.ok:
            print("Image uploaded successfully")
            prediction = response.json()
            print(prediction)
            annotation = ImageAnnotation(prediction["label"], prediction["bbox_xyxy"])
            return annotation
        else:
            raise RuntimeError("error when making prediction")


class ImageReadingWorker(threading.Thread):
    def __init__(
        self, image_reader: ImageReader, image_data_queue: ClosableQueue[ImageData]
    ) -> None:
        super().__init__()

        self.image_reader = image_reader
        self.image_data_queue = image_data_queue

    def run(self) -> None:
        for image in self.image_reader:
            image_data = ImageData(image_orig=image)
            self.image_data_queue.put(image_data)
        self.image_data_queue.close()


class ImagePreprocessingWorker(threading.Thread):
    def __init__(
        self,
        image_preprocessor: ImagePreprocessor,
        image_data_in_queue: ClosableQueue[ImageData],
        image_data_out_queue: ClosableQueue[ImageData],
    ) -> None:
        super().__init__()

        self.image_preprocessor = image_preprocessor
        self.image_data_in_queue = image_data_in_queue
        self.image_data_out_queue = image_data_out_queue

    def run(self) -> None:
        for image_data in self.image_data_in_queue:
            image_data.image_preprocessed = self.image_preprocessor(image_data.image_orig)
            self.image_data_out_queue.put(image_data)
        self.image_data_out_queue.close()


class ImagePredictionWorker(threading.Thread):
    def __init__(
        self,
        image_predictor,
        image_data_in_queue: ClosableQueue[ImageData],
        image_data_out_queue: ClosableQueue[ImageData],
    ) -> None:
        super().__init__()

        self.image_predictor = image_predictor
        self.image_data_in_queue = image_data_in_queue
        self.image_data_out_queue = image_data_out_queue

    def run(self) -> None:
        for image_data in self.image_data_in_queue:
            image_data.annotation = self.image_predictor(image_data.image_orig)
            self.image_data_out_queue.put(image_data)
        self.image_data_out_queue.close()


def square_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        diff_half = (height - width) // 2
        image_cropped = image[diff_half : width + diff_half, :, :]
    else:
        diff_half = (width - height) // 2
        image_cropped = image[:, diff_half : height + diff_half, :]
    return image_cropped


# class ImageRecognitionServiceClientApp(Application):
#     def add_custom_parser_arguments(self, parser: argparse.ArgumentParser) -> None:
#         pass

def main() -> None:
    image_reader = DirectoryImageReader("card_images")
    image_preprocessor = ImagePreprocessor((224, 224))
    image_predictor = ImagePredictor("http://127.0.0.1:5000/predict-image")

    for image in image_reader:
        image_preprocessed = image_preprocessor(image)
        annotation = image_predictor(image_preprocessed)
        print(annotation)


# def main() -> None:
#     input_queue = ClosableQueue()
#     preprocessing_queue = ClosableQueue()
#     prediction_queue = ClosableQueue()
#     # visualization_queue = ClosableQueue()

#     image_reader = DirectoryImageReader("card_images")
#     image_preprocessor = ImagePreprocessor((224, 224))
#     image_predictor = ImagePredictor("http://172.17.0.2:5000/predict-image")

#     input_worker = ImageReadingWorker(image_reader, input_queue)
#     preprocessing_worker = ImagePreprocessingWorker(
#         image_preprocessor, input_queue, preprocessing_queue
#     )
#     prediction_worker = ImagePredictionWorker(
#         image_predictor, preprocessing_queue, prediction_queue
#     )

#     threads = [input_worker, preprocessing_worker, prediction_worker]
#     for thread in threads:
#         thread.start()

#     queues = [input_queue, preprocessing_queue, prediction_queue]
#     for queue in queues:
#         queue.join()

#     for thread in threads:
#         thread.join()

#     for prediction in prediction_queue:
#         print(prediction)


# url = "http://172.17.0.2:5000/predict-image"
# url = "http://127.0.0.1:5000/predict-image"
# image_path = "epidemic.jpg"

# with open(image_path, "rb") as file:
#     response = requests.post(url, files={"image": file})

# if response.ok:
#     print("Image uploaded successfully")
#     print(response.json())
# else:
#     print("Error when processing image")

if __name__ == "__main__":
    main()
    # ImageRecognitionServiceClientApp(
    #     __file__,
    #     description="Predictor that accesses a REST service for localization and classification of Pandemic cards in a given image.",
    # )()
