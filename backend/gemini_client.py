from __future__ import annotations

from typing import Dict, Optional

import google.generativeai as genai

try:  # Prefer typed enums when available
    from google.generativeai import types as genai_types

    DEFAULT_SAFETY_SETTINGS = [
        {
            "category": genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE,
        },
    ]
except Exception:  # pragma: no cover - fallback for older SDKs
    DEFAULT_SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

from .config import Settings


class GeminiClient:
    """Thin wrapper around Gemini SDK with model caching and safety settings."""

    def __init__(self, settings: Settings) -> None:
        """Purpose: Configure the Gemini SDK and initialize model cache.
        Inputs/Outputs: Input is Settings; no return value.
        Side Effects / State: Configures SDK global API key and caches model instances.
        Dependencies: Uses google.generativeai and Settings from config.
        Failure Modes: Raises ValueError if API key or model name is missing.
        If Removed: LLM calls in the pipeline cannot execute and app fails at startup.
        Testing Notes: Validate missing key raises ValueError and models are cached.
        """
        # Configure API key and seed default model cache.
        self._settings = settings
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        genai.configure(api_key=settings.gemini_api_key)
        self._models: Dict[str, genai.GenerativeModel] = {}
        self._default_model = _normalize_model_name(settings.gemini_model_flash)
        if self._default_model:
            self._models[self._default_model] = genai.GenerativeModel(self._default_model)

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 10024,
    ) -> str:
        """Purpose: Generate a single text response from a string prompt.
        Inputs/Outputs: Input is prompt string and optional model/config; returns text.
        Side Effects / State: May add a model to the internal cache.
        Dependencies: Uses genai.GenerativeModel.generate_content.
        Failure Modes: Raises ValueError if model name is missing.
        If Removed: Intent detection and prompt-based steps cannot call the LLM.
        Testing Notes: Ensure non-empty output for valid prompt and model.
        """
        # Resolve model name and ensure cached model instance exists.
        model_name = _normalize_model_name(model) if model else self._default_model
        if not model_name:
            raise ValueError("Gemini model name is required")
        if model_name not in self._models:
            self._models[model_name] = genai.GenerativeModel(model_name)
        response = self._models[model_name].generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            safety_settings=DEFAULT_SAFETY_SETTINGS,
        )
        text: Optional[str] = getattr(response, "text", None)
        return (text or "").strip()

    def generate_content(
        self,
        contents: list,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 10024,
    ) -> str:
        """Purpose: Generate a response from structured chat contents.
        Inputs/Outputs: Input is list of content entries and optional system prompt; returns text.
        Side Effects / State: May add a model to the internal cache.
        Dependencies: Uses genai.GenerativeModel.generate_content and _flatten_contents fallback.
        Failure Modes: Raises ValueError if model name is missing; falls back on TypeError.
        If Removed: Multi-turn LLM generation in the pipeline stops working.
        Testing Notes: Test both structured contents and fallback path for older SDKs.
        """
        # Resolve model name and prepare a cached model instance.
        model_name = _normalize_model_name(model) if model else self._default_model
        if not model_name:
            raise ValueError("Gemini model name is required")
        if model_name not in self._models:
            self._models[model_name] = genai.GenerativeModel(model_name)

        kwargs = {
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            "safety_settings": DEFAULT_SAFETY_SETTINGS,
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        try:
            response = self._models[model_name].generate_content(contents, **kwargs)
        except TypeError:
            if system_instruction:
                combined = f"{system_instruction}\n\n" + _flatten_contents(contents)
            else:
                combined = _flatten_contents(contents)
            fallback_kwargs = {
                "generation_config": kwargs["generation_config"],
                "safety_settings": DEFAULT_SAFETY_SETTINGS,
            }
            response = self._models[model_name].generate_content(combined, **fallback_kwargs)

        text: Optional[str] = getattr(response, "text", None)
        return (text or "").strip()


def _normalize_model_name(name: Optional[str]) -> str:
    """Purpose: Normalize model names by stripping prefix and whitespace.
    Inputs/Outputs: Input is a model name string; output is normalized name.
    Side Effects / State: None.
    Dependencies: None; used by GeminiClient.
    Failure Modes: Returns empty string for falsy input.
    If Removed: Model caching and selection may use invalid names and fail.
    Testing Notes: Ensure "models/foo" becomes "foo" and whitespace is trimmed.
    """
    # Strip "models/" prefix and whitespace.
    if not name:
        return ""
    cleaned = name.strip()
    if cleaned.startswith("models/"):
        return cleaned.split("/", 1)[1]
    return cleaned


def _flatten_contents(contents: list) -> str:
    """Purpose: Convert structured contents into a plain text prompt.
    Inputs/Outputs: Input is a list of content dicts; output is combined text.
    Side Effects / State: None.
    Dependencies: Used by GeminiClient when system_instruction is unsupported.
    Failure Modes: Non-dict entries are skipped; returns empty string if no text parts.
    If Removed: Fallback path for older SDKs fails and raises TypeError.
    Testing Notes: Verify roles are prefixed and parts are concatenated correctly.
    """
    # Flatten role-tagged parts into a readable plain-text prompt.
    parts: list[str] = []
    for entry in contents:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        segments = entry.get("parts", []) or []
        texts = []
        for segment in segments:
            if isinstance(segment, dict):
                text = segment.get("text")
                if text:
                    texts.append(str(text))
        if texts:
            prefix = f"{role.upper()}: " if role else ""
            parts.append(prefix + "\n".join(texts))
    return "\n\n".join(parts)
