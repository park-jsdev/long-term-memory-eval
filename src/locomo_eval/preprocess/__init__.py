"""HLD (i) conversation preprocessing.

DataIngestor wraps dataset.py (no second parser, no source-JSON rewrite).
PreprocessingPipeline runs three deterministic steps in order:
  segment_sessions → assign_turn_ids → normalize_speakers_and_times

CLI: ``python -m src.locomo_eval.preprocess.run_index`` writes
``experiments/<run_id>/preprocess/`` (session blocks + session documents, no LLM).
``raw_chunks`` / ``session_summaries`` may load that dump (``--preprocess-index-run-id``)
or still build from Conversation.
"""

from .conversation_log import SCHEMA_VERSION, write_conversation_run_log
from .data_ingestor import DataIngestor
from .dump import MissingPreprocessIndexError, sample_complete, write_sample_dump
from .preprocessing_pipeline import (
    PreprocessingPipeline,
    assign_turn_ids,
    normalize_speakers_and_times,
    parse_session_datetime,
    segment_sessions,
)
from .session_documents import (
    build_session_documents,
    join_qa_to_documents,
    naive_rank_session_ids,
)

__all__ = [
    "SCHEMA_VERSION",
    "DataIngestor",
    "MissingPreprocessIndexError",
    "PreprocessingPipeline",
    "assign_turn_ids",
    "build_session_documents",
    "join_qa_to_documents",
    "naive_rank_session_ids",
    "normalize_speakers_and_times",
    "parse_session_datetime",
    "sample_complete",
    "segment_sessions",
    "write_conversation_run_log",
    "write_sample_dump",
]
