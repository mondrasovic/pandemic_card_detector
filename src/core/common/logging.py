from __future__ import annotations

import logging

__first_call = True


def get_logger(module_name: str) -> logging.Logger:
    global __first_call

    if __first_call:
        logging.basicConfig(
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        __first_call = False

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    return logger
