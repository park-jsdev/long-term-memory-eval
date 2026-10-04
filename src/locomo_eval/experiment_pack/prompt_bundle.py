"""Snapshot configs + prompts into a run pack so a human can walk YAML → txt → jsonl.

Written by ``run.py`` (QA) and merged by the autorater CLI. Analysis should
read ``TRACE.md`` / ``prompts/index.json``; it does not need this module.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.config import ROOT, list_config_chain
from src.locomo_eval.prompts import locate_prompt_file
from src.locomo_eval.report import write_json

from .audit_layout import AUDIT_LAYOUT_VERSION, AuditPaths

PROMPT_BUNDLE_SCHEMA = "prompt_bundle.v1"

_PROMPT_KEYS = frozenset({"prompt_path", "graph_prompt_path"})


def repo_rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve() if candidate.exists() else candidate
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix().replace("\\", "/")


def walk_prompt_fields(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Return (dotted config key, prompt path string) for every prompt field."""
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key in _PROMPT_KEYS or str(key).endswith("_prompt_path"):
                if value:
                    found.append((dotted, str(value)))
            else:
                found.extend(walk_prompt_fields(value, dotted))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            found.extend(walk_prompt_fields(item, f"{prefix}[{index}]"))
    return found


def role_for(config_key: str, source: str) -> str:
    key = config_key.replace("\\", "/")
    src = source.replace("\\", "/")
    if "autorater" in key or "/autoraters/" in src:
        return "autorater"
    if "/agents/" in src or key.startswith("agent."):
        return "agent"
    if key.endswith("pipeline.prompt_path") or "/readers/" in src:
        return "reader"
    if "writer" in key or "/writers/" in src:
        return "writer"
    return "other"


def jsonl_targets(config_key: str, cfg: dict[str, Any]) -> list[str]:
    key = config_key
    if "autorater" in key:
        return ["autorater/traces.jsonl", "autorater/autorater_verdicts.jsonl"]
    if key.endswith("pipeline.prompt_path"):
        targets = [
            "predictions.jsonl",
            "reader/traces.jsonl",
            "reader/predictions.jsonl",
        ]
        if "/agents/" in str((cfg.get("pipeline") or {}).get("prompt_path") or ""):
            targets.extend(
                [
                    "agent/traces.jsonl",
                    "agent/trajectory.jsonl",
                ]
            )
        return targets
    if "writer" in key or "graph_prompt" in key:
        return ["memory/writer/calls.jsonl", "memory/writer/index.jsonl"]
    if "openai_memory" in key:
        index_run = (cfg.get("openai_memory") or {}).get("index_run_id")
        if index_run:
            return [f"experiments/{index_run}/openai_memory_index/"]
        return ["experiments/<openai_memory.index_run_id>/openai_memory_index/"]
    if "mem0" in key:
        index_run = (cfg.get("mem0") or {}).get("index_run_id")
        if index_run:
            return [f"experiments/{index_run}/mem0_index/"]
        return ["experiments/<mem0.index_run_id>/mem0_index/"]
    return []


def _sha256_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot_rel(source: Path) -> str:
    try:
        rel = source.resolve().relative_to(ROOT.resolve())
        if rel.parts and rel.parts[0] == "prompts":
            return rel.as_posix()
    except ValueError:
        pass
    return f"prompts/{source.name}"


def collect_prompt_rows(
    cfg: dict[str, Any],
    extra: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    fields = walk_prompt_fields(cfg)
    if extra:
        fields.extend(extra)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for config_key, raw_path in fields:
        marker = (config_key, raw_path)
        if marker in seen:
            continue
        seen.add(marker)
        source = locate_prompt_file(raw_path)
        rows.append(
            {
                "role": role_for(config_key, repo_rel(source)),
                "config_key": config_key,
                "source_path": repo_rel(source),
                "snapshot_path": _snapshot_rel(source),
                "prompt_version": source.stem,
                "sha256": _sha256_text(source),
                "jsonl": jsonl_targets(config_key, cfg),
            }
        )
    return rows


def write_prompt_bundle(
    run_dir: Path,
    cfg: dict[str, Any],
    *,
    config_entry: str | Path | None = None,
    extra_prompts: list[tuple[str, str]] | None = None,
    merge: bool = False,
) -> Path:
    """Copy used prompts into ``experiments/<run_id>/prompts/`` and write TRACE.md."""
    paths = AuditPaths.from_run_dir(run_dir)
    rows = collect_prompt_rows(cfg, extra=extra_prompts)
    existing: dict[str, Any] = {}
    if merge and paths.prompt_index.is_file():
        existing = json.loads(paths.prompt_index.read_text(encoding="utf-8")) or {}
        prior = {row["config_key"]: row for row in existing.get("prompts") or []}
        for row in rows:
            prior[row["config_key"]] = row
        rows = list(prior.values())

    paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = run_dir / row["snapshot_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(locate_prompt_file(row["source_path"]), dest)

    includes: list[str] = []
    entry_rel = None
    if config_entry:
        entry_path = Path(config_entry)
        if not entry_path.is_file():
            entry_path = ROOT / entry_path
        if entry_path.is_file():
            entry_rel = repo_rel(entry_path)
            includes = [repo_rel(p) for p in list_config_chain(entry_path)]
            # write_frozen_config already copied the entry when the QA pipeline ran.
            if not merge and not paths.config_source.is_file():
                shutil.copy2(entry_path, paths.config_source)

    if merge:
        includes = existing.get("config_includes") or includes
        entry_rel = existing.get("config_entry") or entry_rel

    # Claim-audit frozen config includes CLI overrides; do not replace it with YAML-only.
    if not merge and not paths.config_resolved.is_file():
        with paths.config_resolved.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                cfg,
                handle,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    payload = {
        "schema_version": PROMPT_BUNDLE_SCHEMA,
        "audit_layout": AUDIT_LAYOUT_VERSION,
        "config_entry": entry_rel or existing.get("config_entry"),
        "config_includes": includes or existing.get("config_includes") or [],
        "prompts": rows,
    }
    write_json(paths.prompt_index, payload)
    paths.trace.write_text(_render_trace(payload, run_dir=run_dir), encoding="utf-8")
    return paths.trace


def find_run_dir_for_autorater(out_dir: str | Path) -> Path | None:
    """Walk up from an autorater out dir to the QA pack (has predictions.jsonl).

    ``autorater/run_meta.json`` is the judge pack, not the QA pack, so do not
    treat a ``run_meta.json`` alone as the sandwich run root.
    """
    current = Path(out_dir).resolve()
    for _ in range(4):
        if (current / "predictions.jsonl").is_file() or (current / "TRACE.md").is_file():
            if current.name != "autorater":
                return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _render_trace(payload: dict[str, Any], *, run_dir: Path) -> str:
    lines = [
        f"# Trace `{run_dir.name}`",
        "",
        "Walk **config YAML → prompt txt → jsonl**. Snapshot copies live under `prompts/`.",
        "",
        "## 1. Configs",
        "",
    ]
    includes = payload.get("config_includes") or []
    entry = payload.get("config_entry")
    if includes:
        for i, rel in enumerate(includes, start=1):
            mark = " (CLI `--config`)" if rel == entry else ""
            lines.append(f"{i}. `{rel}`{mark}")
        lines.append("")
        lines.append("Resolved merge (later keys win): `config.resolved.yaml`")
        if (run_dir / "config.source.yaml").is_file():
            lines.append("Entry file copy: `config.source.yaml`")
    elif entry:
        lines.append(f"- `{entry}`")
    else:
        lines.append("- In-memory config (no `--config` path). See `config.resolved.yaml`.")
    lines.extend(["", "## 2. Prompts", ""])
    rows = payload.get("prompts") or []
    if not rows:
        lines.append("No prompt_path fields in the resolved config.")
    else:
        lines.extend(
            [
                "| Role | Config key | Source | Snapshot | Version | JSONL |",
                "|------|------------|--------|----------|---------|-------|",
            ]
        )
        for row in rows:
            jsonl = ", ".join(f"`{p}`" for p in row.get("jsonl") or []) or "—"
            lines.append(
                f"| {row['role']} | `{row['config_key']}` | `{row['source_path']}` | "
                f"`{row['snapshot_path']}` | `{row['prompt_version']}` | {jsonl} |"
            )
    lines.extend(
        [
            "",
            "## 3. Outputs in this directory",
            "",
            "| File | What it is |",
            "|------|------------|",
            "| `predictions.jsonl` | One QA row: `{memory}` text, pred, gold (scorer-only) |",
            "| `reader/traces.jsonl` | Answer LLM call (filled reader prompt → completion) |",
            "| `reader/predictions.jsonl` | Same QA rows as the run-root copy |",
            "| `memory/` | `{memory}` payload actually inserted into the reader prompt |",
            "| `memory/writer/calls.jsonl` | Write-path calls (if a writer ran) |",
            "| `autorater/traces.jsonl` | Judge LLM (separate job; gold is visible here) |",
            "| `run_meta.json` | Pins: models, `prompt_path`, hashes |",
            "| `SUMMARY.md` | Claim audit: sandwich pins, cost, how to follow one question |",
            "| `ATTRIBUTION.md` | LLM call → role → claims made |",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
