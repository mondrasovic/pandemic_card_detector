from __future__ import annotations

import bisect
import pathlib
import shutil
import sys
from typing import TYPE_CHECKING

import click
import tqdm

if TYPE_CHECKING:
    from os import PathLike

IMG_ID_CLASS_MAP = (
    (51, "epidemic"),
    (63, "town_yellow"),
    (75, "town_blue"),
    (87, "town_black"),
    (99, "town_red"),
    (111, "infection_blue"),
    (123, "infection_yellow"),
    (135, "infection_black"),
    (147, "infection_red"),
    (162, "event"),
    (172, "emergency_state"),
    (173, "mutation"),
    (183, "bonus_lab"),
    (191, "bonus_emergency_state"),
    (210, "player"),
)


def get_image_class_name(image_file_stem_name: str) -> str:
    image_id = int(image_file_stem_name.split("_")[1])
    class_index = bisect.bisect_left(IMG_ID_CLASS_MAP, (image_id, ""))
    class_name = IMG_ID_CLASS_MAP[class_index][1]

    return class_name


@click.command()
@click.argument("input_dir_path", type=click.Path(exists=True, file_okay=False, readable=True))
@click.argument("output_dir_path", type=click.Path(file_okay=False, writable=True))
def main(input_dir_path: str | PathLike[str], output_dir_path: str | PathLike[str]) -> int:
    input_dir = pathlib.Path(input_dir_path)
    output_dir = pathlib.Path(output_dir_path)

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    images_pbar = tqdm.tqdm(enumerate(input_dir.iterdir()))
    for image_id, input_image_file in images_pbar:
        images_pbar.set_description(f"processing image {input_image_file.stem}")

        class_name = get_image_class_name(input_image_file.stem)

        output_file = output_dir / class_name / f"{image_id:05d}{input_image_file.suffix}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(input_image_file), str(output_file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
