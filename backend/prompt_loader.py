from __future__ import annotations

from pathlib import Path


def load_prompt(prompt_path: Path) -> str:
    """Purpose: Load a prompt file as UTF-8 text and strip BOM if present.
    Inputs/Outputs: Input is a Path to the prompt file; output is the decoded string.
    Side Effects / State: None; pure function reading the filesystem.
    Dependencies: Uses Path.read_text/read_bytes; used by pipeline generation steps.
    Failure Modes: UnicodeDecodeError triggers a fallback decode with errors ignored,
        which can drop invalid bytes.
    If Removed: Prompt loading fails and LLM calls in intent/generation will crash.
    Testing Notes: Validate BOM-stripping and fallback decoding on non-UTF8 files.
    """
    # Read as UTF-8 and fall back to a tolerant decode if needed.
    try:
        return prompt_path.read_text(encoding="utf-8").lstrip("\ufeff")
    except UnicodeDecodeError:
        raw = prompt_path.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
        return text.lstrip("\ufeff")
