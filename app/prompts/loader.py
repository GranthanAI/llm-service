"""
Prompt Template YAML Loader.
Implements LLD v2.0 Section 17.1.
"""

from pathlib import Path

import structlog
import yaml

from app.config.logging import get_logger
from app.prompts.schemas import PromptTemplateConfig

logger = get_logger("prompt_loader")


class PromptLoader:
    """
    Loads and validates prompt YAML templates from disk.
    Supports mode-level templates and graph node fragments.
    """

    def __init__(
        self,
        prompts_dir: Path | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.prompts_dir = prompts_dir or (Path(__file__).parent)
        self.logger = logger_instance or logger

    def load_all(self) -> list[PromptTemplateConfig]:
        """Recursively scan prompts_dir for all .yaml and .yml files and parse them."""
        configs: list[PromptTemplateConfig] = []
        if not self.prompts_dir.exists():
            self.logger.warning("Prompts directory does not exist", path=str(self.prompts_dir))
            return configs

        for path in self.prompts_dir.rglob("*.yaml"):
            try:
                config = self.load_file(path)
                configs.append(config)
            except Exception as e:
                self.logger.error(
                    "Failed to load prompt template file", path=str(path), error=str(e)
                )

        for path in self.prompts_dir.rglob("*.yml"):
            try:
                config = self.load_file(path)
                configs.append(config)
            except Exception as e:
                self.logger.error(
                    "Failed to load prompt template file", path=str(path), error=str(e)
                )

        self.logger.info(
            "Loaded prompt templates", count=len(configs), directory=str(self.prompts_dir)
        )
        return configs

    def load_file(self, path: Path) -> PromptTemplateConfig:
        """Loads and validates a single YAML prompt template file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {path}: expected dictionary root")

        config = PromptTemplateConfig(**data)
        self._validate(config)
        return config

    def _validate(self, config: PromptTemplateConfig) -> None:
        """Validates configuration consistency."""
        if not config.name or not config.system:
            raise ValueError(f"Prompt template must specify 'name' and 'system': {config.name}")
        if not config.mode and not (config.graph and config.node):
            raise ValueError(
                f"Prompt template {config.name} must specify either 'mode' or both ('graph' and 'node')"
            )
