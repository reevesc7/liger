import pytest


pytest.importorskip("openai", reason="openai not installed")
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers not installed",
)
