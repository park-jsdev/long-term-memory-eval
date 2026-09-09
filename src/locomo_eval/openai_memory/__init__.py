"""OpenAI-memory baseline: extract-all dump, retrieve-all (privileged).

The Mem0 paper's OpenAI row used ChatGPT's product Memory UI (no public
selective-retrieve API). This package is an **architecture clone** of that
privileged-context protocol: extract timestamped entries with an LLM, then
inject *all* of them as ``Memory.text``. It cannot reproduce ChatGPT Memory
and is not a Table 2 J=52.90 claim.
"""

from .builders import OpenAIMemoryBuilder
from .extract import MockOpenAIMemoryExtractor, get_openai_memory_extractor

SCHEMA_VERSION = "openai_memory_index.v1"

__all__ = [
    "SCHEMA_VERSION",
    "MockOpenAIMemoryExtractor",
    "OpenAIMemoryBuilder",
    "get_openai_memory_extractor",
]
