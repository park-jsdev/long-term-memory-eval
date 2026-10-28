"""Personalization records, decisions, and pack diagnostics.

Pipeline step after LoCoMo load and before the Codex hook. Annotation
and representation stay here so H2 can be scored with no agent process.
The marker hook only delivers the policy file this package writes.
"""
