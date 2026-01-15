from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Configuration container for models, resources, and runtime limits."""
    gemini_api_key: str
    gemini_model_flash: str
    gemini_model_pro: str
    resources_path: Path
    prompts_dir: Path
    max_images: int
    max_attempts: int


def load_settings() -> Settings:
    """Purpose: Load configuration from environment variables and defaults.
    Inputs/Outputs: No inputs; returns a Settings instance.
    Side Effects / State: Reads environment variables and filesystem paths.
    Dependencies: Uses os.getenv and BASE_DIR for default paths.
    Failure Modes: Invalid MAX_IMAGES/MAX_ATTEMPTS env values raise ValueError.
    If Removed: App cannot configure models/resources and fails at startup.
    Testing Notes: Verify defaults and overrides via environment variables.
    """
    # Resolve resource and prompt paths, then build Settings.
    resources_path = os.getenv("RESOURCES_PATH")
    if resources_path:
        resources_file = Path(resources_path)
    else:
        resources_file = (BASE_DIR / ".." / "resources" / "AgentX.json").resolve()

    prompts_dir = (BASE_DIR / "prompts").resolve()

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model_flash=os.getenv("GEMINI_MODEL_FLASH")
        or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_model_pro=os.getenv("GEMINI_MODEL_PRO")
        or os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        resources_path=resources_file,
        prompts_dir=prompts_dir,
        max_images=int(os.getenv("MAX_IMAGES", "4")),
        max_attempts=int(os.getenv("MAX_ATTEMPTS", "3")),
    )
