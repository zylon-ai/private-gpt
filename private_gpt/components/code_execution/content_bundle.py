"""Compatibility re-export — the models live in the sandbox layer now."""

from private_gpt.components.sandbox.content_bundle import BundledFile, ContentBundle

__all__ = ["BundledFile", "ContentBundle"]
