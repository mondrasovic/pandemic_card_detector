import pathlib
import shutil
import sys

import click
import cv2 as cv
import numpy as np
import pillow_heif
import tqdm
from PIL import Image

MAX_SIDE_SIZE = 960
IMAGE_TYPE = "jpeg"


def get_output_file_path(output_dir, input_file, image_type=IMAGE_TYPE):
    return output_dir / f"{input_file.stem}.{image_type}"


def read_image(image_file):
    image = Image.open(image_file)
    image_arr = np.asarray(image)
    image_arr = cv.cvtColor(image_arr, cv.COLOR_RGB2BGR)

    return image_arr


def resize_image(image, max_side_size=MAX_SIDE_SIZE):
    height, width = image.shape[:2]

    scale = None
    if height > max_side_size:
        scale = max_side_size / height
    if width > max_side_size:
        scale = min(1 if scale is None else scale, max_side_size / width)

    if scale is None:
        image_out = image
    else:
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        image_out = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)

    return image_out


def denoise(image):
    return cv.fastNlMeansDenoisingColored(image, h=10)


def threshold(image, block_size=15, const=5):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_thresh = cv.adaptiveThreshold(
        image_gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        blockSize=block_size,
        C=const,
    )

    return image_thresh


def dilate(image, kernel_size=11, n_iters=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_dilated = cv.dilate(image, kernel, iterations=n_iters)

    return image_dilated


def find_biggest_roi_bbox(image, area_min_ratio=0.05):
    contours, _ = cv.findContours(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    max_area = 0
    max_area_bbox = None

    for contour in contours:
        contour_area = cv.contourArea(contour)
        if (contour_area / image_area) < area_min_ratio:
            continue

        if contour_area > max_area:
            bbox = cv.boundingRect(contour)
            max_area, max_area_bbox = contour_area, bbox

    return max_area_bbox


def scale_bbox(bbox, x_scale=1, y_scale=1):
    x, y, width, height = bbox
    width_half, height_half = width / 2, height / 2

    x_center = x + width_half
    y_center = y + height_half

    new_width = width * x_scale
    new_height = height * y_scale

    new_x = int(round(x_center - (new_width / 2)))
    new_y = int(round(y_center - (new_height / 2)))
    new_width = int(round(new_width))
    new_height = int(round(new_height))

    return new_x, new_y, new_width, new_height


def extract_card_from_image(input_image_file, output_image_file):
    image = read_image(input_image_file)
    image_resized = resize_image(image)

    image_denoised = denoise(image_resized)
    image_thresh = threshold(image_denoised)
    image_dilated = dilate(image_thresh)
    bbox = find_biggest_roi_bbox(image_dilated)
    if bbox is None:
        print(
            f"Error: no bounding box found for {input_image_file.stem}.",
            file=sys.stderr,
        )
        return

    scale = 0.92
    bbox = scale_bbox(bbox, x_scale=scale, y_scale=scale)

    x, y, width, height = bbox
    card_roi = image_resized[y : y + height, x : x + width]

    cv.imwrite(str(output_image_file), card_roi)


@click.command()
@click.argument("input_dir_path", type=click.Path(exists=True))
@click.argument("output_dir_path", type=click.Path())
def main(input_dir_path, output_dir_path):
    pillow_heif.register_heif_opener()

    input_dir = pathlib.Path(input_dir_path)
    output_dir = pathlib.Path(output_dir_path)

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    images_pbar = tqdm.tqdm(input_dir.iterdir())
    for input_image_file in images_pbar:
        images_pbar.set_description(f"processing {input_image_file.stem}")
        output_image_file = get_output_file_path(output_dir, input_image_file)
        extract_card_from_image(input_image_file, output_image_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
