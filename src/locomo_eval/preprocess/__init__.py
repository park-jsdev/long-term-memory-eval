"""HLD (i) conversation preprocessing.

DataIngestor wraps dataset.py (no second parser, no source-JSON rewrite).
PreprocessingPipeline runs three deterministic steps in order:
  segment_sessions → assign_turn_ids → normalize_speakers_and_times

Not wired into run.py; C0/C1 still consume Conversation.
"""

from .conversation_log import SCHEMA_VERSION, write_conversation_run_log
from .data_ingestor import DataIngestor
from .preprocessing_pipeline import (
    PreprocessingPipeline,
    assign_turn_ids,
    normalize_speakers_and_times,
    parse_session_datetime,
    segment_sessions,
)

__all__ = [
    "SCHEMA_VERSION",
    "DataIngestor",
    "PreprocessingPipeline",
    "assign_turn_ids",
    "normalize_speakers_and_times",
    "parse_session_datetime",
    "segment_sessions",
    "write_conversation_run_log",
]
