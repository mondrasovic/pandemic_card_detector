from __future__ import annotations

import abc
import argparse
from typing import TYPE_CHECKING

from core.common.config import get_config_defaults_or_override
from core.common.logging import get_logger

if TYPE_CHECKING:
    from typing import Optional

    from yacs.config import CfgNode as ConfigurationNode


class Application:
    def __init__(self, logger_name: str = "app", description: Optional[str] = None) -> None:
        self.logger = get_logger(logger_name)
        self.description = description

    def __call__(self) -> None:
        self.run()

    def run(self) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        self.add_default_parser_arguments(parser)
        self.add_custom_parser_arguments(parser)
        self.args = parser.parse_args()

        # TODO Add overriding configuration from comamnd line arguments
        self.config: ConfigurationNode = get_config_defaults_or_override(self.args.config_file_path)
        self.config.freeze()

        self.main()

    @abc.abstractmethod
    def main(self) -> None:
        pass

    def log_info(self, *args, **kwargs) -> None:
        self.logger.info(*args, **kwargs)

    def log_debug(self, *args, **kwargs) -> None:
        self.logger.debug(*args, **kwargs)

    def log_warning(self, *args, **kwargs) -> None:
        self.logger.warning(*args, **kwargs)

    def log_error(self, *args, **kwargs) -> None:
        self.logger.error(*args, **kwargs)

    def add_default_parser_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-c",
            "--config",
            dest="config_file_path",
            type=str,
            default=None,
            help="YAML configuration file path to overwrite the default values",
        )

    def add_custom_parser_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass
