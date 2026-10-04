"""Deterministic sandwich-pack verifier (configs, prompts, schemas, logs).

No LLM. Reads a finished ``experiments/<run_id>/`` (or a thin aggregate
catalog copy) and returns ordered checks. Analysis/CLI import this module;
do not import ``run``, the writer model, or ``audit_writer``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.locomo_eval.experiment_pack.audit_layout import AUDIT_LAYOUT_VERSION, AuditPaths
from src.locomo_eval.experiment_pack.audit_loader import load_json, load_jsonl, predictions_jsonl
from src.locomo_eval.prompts import locate_prompt_file

SCHEMA_VERSION = "experiment_verify.v1"
QA_MEM0_V1 = "prompts/readers/qa_mem0_v1.txt"
QA_MEM0_V1_SHA256 = "85c626a7eaf0631e17d3fcdaa020e47afb2d85802f1d4543d6363f812fbfe31c"
GRAPH_PROMPT = "prompts/writers/graph_v1.txt"
GRAPH_METHODS = frozenset({"graph"})
SANDWICH_READER = "gpt-4o-mini"
PARSE_FALLBACK_MAX = 0.20
GOLD_MIN_CHARS = 12
DATE_TOKEN_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    re.I,
)
SPEAKER_LINE_RE = re.compile(r"\|\s*[A-Za-z][^|]{0,40}:\s")

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass(frozen=True)
class Check:
    """One verifier assertion. ``id`` is stable for diffs across machines."""

    id: str
    layer: str
    severity: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PackReport:
    """Verification of one run directory."""

    schema_version: str
    run_dir: str
    run_id: str
    memory_type: str | None
    pack_kind: str
    verdict: str
    checks: list[Check]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "memory_type": self.memory_type,
            "pack_kind": self.pack_kind,
            "verdict": self.verdict,
            "n_fail": n_status(self.checks, STATUS_FAIL),
            "n_warn_fail": n_warning_fails(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


def n_status(checks: list[Check], status: str) -> int:
    return sum(1 for c in checks if c.status == status)


def n_warning_fails(checks: list[Check]) -> int:
    return sum(
        1
        for c in checks
        if c.status == STATUS_FAIL and c.severity == SEVERITY_WARNING
    )


def verdict_for(checks: list[Check], pack_kind: str) -> str:
    errors = [
        c
        for c in checks
        if c.status == STATUS_FAIL and c.severity == SEVERITY_ERROR
    ]
    if errors:
        return "invalid"
    if pack_kind == "incomplete":
        return "incomplete"
    warnings = [
        c
        for c in checks
        if c.status == STATUS_FAIL and c.severity == SEVERITY_WARNING
    ]
    if warnings:
        return "valid_with_warnings"
    return "valid"


def classify_pack_kind(paths: AuditPaths, meta: dict[str, Any]) -> str:
    """full_audit has memory dumps; thin_catalog is aggregate/by_run; else incomplete."""
    if not paths.run_meta.is_file():
        return "incomplete"
    has_preds = predictions_jsonl(paths.run_dir) is not None
    has_metrics = paths.metrics.is_file()
    has_memory = (paths.memory_dir / "index.jsonl").is_file() or (
        paths.memory_dir / "schema.json"
    ).is_file()
    if has_memory and (has_preds or has_metrics):
        return "full_audit"
    if has_metrics and meta:
        return "thin_catalog"
    if has_preds or has_metrics:
        return "incomplete"
    return "incomplete"


def verify_pack(run_dir: str | Path) -> PackReport:
    """Walk config → prompt snapshot → jsonl logs for one sandwich run."""
    paths = AuditPaths.from_run_dir(run_dir)
    meta = load_json(paths.run_meta)
    memory_type = str(meta.get("memory_type") or "") or None
    kind = classify_pack_kind(paths, meta)
    checks: list[Check] = []
    checks.extend(_layout_checks(paths, meta, kind))
    checks.extend(_config_checks(paths, meta, memory_type))
    checks.extend(_prompt_checks(paths, memory_type))
    checks.extend(_memory_checks(paths, meta, memory_type, kind))
    checks.extend(_graph_checks(paths, memory_type, kind))
    checks.extend(_gold_checks(paths, kind))
    checks.sort(key=lambda c: c.id)
    return PackReport(
        schema_version=SCHEMA_VERSION,
        run_dir=str(paths.run_dir),
        run_id=str(meta.get("run_id") or paths.run_dir.name),
        memory_type=memory_type,
        pack_kind=kind,
        verdict=verdict_for(checks, kind),
        checks=checks,
    )


def _ok(
    check_id: str,
    layer: str,
    detail: str,
    *,
    severity: str = SEVERITY_ERROR,
    evidence: dict[str, Any] | None = None,
) -> Check:
    return Check(check_id, layer, severity, STATUS_PASS, detail, evidence or {})


def _fail(
    check_id: str,
    layer: str,
    detail: str,
    *,
    severity: str = SEVERITY_ERROR,
    evidence: dict[str, Any] | None = None,
) -> Check:
    return Check(check_id, layer, severity, STATUS_FAIL, detail, evidence or {})


def _skip(
    check_id: str,
    layer: str,
    detail: str,
    *,
    severity: str = SEVERITY_INFO,
    evidence: dict[str, Any] | None = None,
) -> Check:
    return Check(check_id, layer, severity, STATUS_SKIP, detail, evidence or {})


def _layout_checks(
    paths: AuditPaths, meta: dict[str, Any], kind: str
) -> list[Check]:
    out: list[Check] = []
    if not paths.run_meta.is_file():
        return [
            _fail(
                "layout.run_meta",
                "layout",
                f"missing {paths.run_meta.name}",
            )
        ]
    layout = meta.get("audit_layout") or {}
    version = layout.get("version") if isinstance(layout, dict) else None
    if version == AUDIT_LAYOUT_VERSION:
        out.append(
            _ok(
                "layout.audit_version",
                "layout",
                f"audit_layout.version={version}",
                evidence={"version": version},
            )
        )
    elif version is None:
        out.append(
            _fail(
                "layout.audit_version",
                "layout",
                "run_meta.json has no audit_layout.version (pre-v2 pack)",
                severity=SEVERITY_WARNING,
            )
        )
    else:
        out.append(
            _fail(
                "layout.audit_version",
                "layout",
                f"audit_layout.version={version}, expected {AUDIT_LAYOUT_VERSION}",
                severity=SEVERITY_WARNING,
                evidence={"version": version},
            )
        )
    if paths.metrics.is_file():
        out.append(_ok("layout.metrics", "layout", "metrics.json present"))
    else:
        out.append(_fail("layout.metrics", "layout", "missing metrics.json"))
    pred = predictions_jsonl(paths.run_dir)
    n_meta = meta.get("n_predictions")
    if pred is None:
        sev = SEVERITY_WARNING if kind == "thin_catalog" else SEVERITY_ERROR
        out.append(
            _fail(
                "layout.predictions",
                "layout",
                "no predictions.jsonl (thin catalog is ok for score-only cells)",
                severity=sev,
            )
        )
    else:
        n_rows = sum(1 for _ in load_jsonl(pred))
        if n_meta is None or int(n_meta) == n_rows:
            out.append(
                _ok(
                    "layout.predictions",
                    "layout",
                    f"{n_rows} prediction rows",
                    evidence={"n": n_rows, "n_meta": n_meta},
                )
            )
        else:
            out.append(
                _fail(
                    "layout.predictions",
                    "layout",
                    f"n_predictions={n_meta} but jsonl has {n_rows} rows",
                    evidence={"n": n_rows, "n_meta": n_meta},
                )
            )
    if paths.trace.is_file():
        out.append(_ok("layout.trace", "layout", "TRACE.md present"))
    else:
        out.append(
            _fail(
                "layout.trace",
                "layout",
                "missing TRACE.md",
                severity=SEVERITY_WARNING,
            )
        )
    if paths.config_source.is_file() and paths.config_resolved.is_file():
        out.append(
            _ok(
                "layout.frozen_config",
                "layout",
                "config.source.yaml and config.resolved.yaml present",
            )
        )
    else:
        out.append(
            _fail(
                "layout.frozen_config",
                "layout",
                "missing config.source.yaml and/or config.resolved.yaml",
                severity=SEVERITY_WARNING,
            )
        )
    if paths.cost.is_file():
        out.append(_ok("layout.cost", "layout", "cost.json present"))
    else:
        out.append(
            _fail(
                "layout.cost",
                "layout",
                "missing cost.json",
                severity=SEVERITY_WARNING,
            )
        )
    return out


def _config_checks(
    paths: AuditPaths, meta: dict[str, Any], memory_type: str | None
) -> list[Check]:
    out: list[Check] = []
    reader = str(meta.get("reader_model") or "")
    prompt = str(meta.get("prompt_path") or "").replace("\\", "/")
    layout = str(meta.get("message_layout") or "")
    if memory_type:
        out.append(
            _ok(
                "config.memory_type",
                "config",
                f"memory_type={memory_type}",
                evidence={"memory_type": memory_type},
            )
        )
    else:
        out.append(
            _fail("config.memory_type", "config", "run_meta missing memory_type")
        )
    sandwichish = "mem0-reader" in paths.run_dir.name or str(
        meta.get("run_id") or ""
    ).startswith("locomo-mem0-reader")
    if sandwichish:
        if reader == SANDWICH_READER:
            out.append(
                _ok(
                    "config.sandwich_reader",
                    "config",
                    f"frozen reader {reader}",
                    evidence={"reader_model": reader},
                )
            )
        else:
            out.append(
                _fail(
                    "config.sandwich_reader",
                    "config",
                    f"sandwich pack reader={reader!r}, expected {SANDWICH_READER}",
                    evidence={"reader_model": reader},
                )
            )
    else:
        out.append(
            _skip(
                "config.sandwich_reader",
                "config",
                "not a mem0-reader sandwich pack",
                evidence={"reader_model": reader},
            )
        )
    prompt_norm = prompt.replace("prompts/", "")
    if prompt.endswith("qa_mem0_v1.txt") or prompt_norm.endswith("qa_mem0_v1.txt"):
        out.append(
            _ok(
                "config.reader_prompt",
                "config",
                f"prompt_path={prompt}",
                evidence={"prompt_path": prompt},
            )
        )
    else:
        out.append(
            _fail(
                "config.reader_prompt",
                "config",
                f"prompt_path={prompt!r} is not qa_mem0_v1 (Mem0-parity freeze)",
                severity=SEVERITY_WARNING,
                evidence={"prompt_path": prompt},
            )
        )
    if layout == "mem0_system_only":
        out.append(
            _ok(
                "config.message_layout",
                "config",
                "mem0_system_only (released Mem0 Chat Completions layout)",
            )
        )
    elif layout == "default_system_user":
        out.append(
            _fail(
                "config.message_layout",
                "config",
                "message_layout=default_system_user, not mem0_system_only "
                "(qa_default stack). Same across writer cells, so it does not "
                "explain graph vs session ranking, but it is not Mem0-parity.",
                severity=SEVERITY_WARNING,
                evidence={"message_layout": layout},
            )
        )
    else:
        out.append(
            _skip(
                "config.message_layout",
                "config",
                f"message_layout={layout or 'unset'}",
            )
        )
    return out


def _prompt_checks(paths: AuditPaths, memory_type: str | None) -> list[Check]:
    out: list[Check] = []
    index = load_json(paths.prompt_index)
    rows = list(index.get("prompts") or []) if index else []
    if not rows:
        return [
            _fail(
                "prompt.bundle",
                "prompt",
                "missing prompts/index.json",
                severity=SEVERITY_WARNING,
            )
        ]
    out.append(
        _ok("prompt.bundle", "prompt", f"{len(rows)} snapshot prompt(s)")
    )
    for row in sorted(rows, key=lambda r: str(r.get("config_key") or "")):
        snap_rel = str(row.get("snapshot_path") or "")
        snap = paths.run_dir / snap_rel if snap_rel else None
        recorded = str(row.get("sha256") or "")
        if snap is None or not snap.is_file():
            out.append(
                _fail(
                    f"prompt.snapshot:{row.get('config_key')}",
                    "prompt",
                    f"missing snapshot {snap_rel}",
                    severity=SEVERITY_WARNING,
                )
            )
            continue
        digest = _sha256_file(snap)
        key = str(row.get("config_key") or snap_rel)
        if recorded and digest != recorded:
            out.append(
                _fail(
                    f"prompt.hash_internal:{key}",
                    "prompt",
                    "snapshot bytes do not match prompts/index.json sha256",
                    evidence={"sha256": digest, "recorded": recorded},
                )
            )
        else:
            out.append(
                _ok(
                    f"prompt.hash_internal:{key}",
                    "prompt",
                    f"{snap_rel} matches index sha256",
                )
            )
        source = str(row.get("source_path") or "")
        try:
            repo_file = locate_prompt_file(source)
        except FileNotFoundError:
            repo_file = None
        if repo_file is not None and repo_file.is_file():
            repo_hash = _sha256_file(repo_file)
            if repo_hash != digest:
                out.append(
                    _fail(
                        f"prompt.hash_repo:{key}",
                        "prompt",
                        f"run snapshot drifted from current repo {source}",
                        severity=SEVERITY_WARNING,
                        evidence={"snapshot": digest, "repo": repo_hash},
                    )
                )
            else:
                out.append(
                    _ok(
                        f"prompt.hash_repo:{key}",
                        "prompt",
                        f"snapshot matches current {source}",
                    )
                )
    if memory_type in GRAPH_METHODS:
        graph_rows = [
            r
            for r in rows
            if "graph_prompt" in str(r.get("config_key") or "")
            or str(r.get("source_path") or "").replace("\\", "/").endswith(
                "graph_v1.txt"
            )
        ]
        if graph_rows:
            text = ""
            rel = str(graph_rows[0].get("snapshot_path") or GRAPH_PROMPT)
            snap = paths.run_dir / rel
            if snap.is_file():
                text = snap.read_text(encoding="utf-8")
            if "timeless" in text.lower():
                out.append(
                    _ok(
                        "prompt.graph_timeless",
                        "prompt",
                        "graph_v1 asks for timeless relation types "
                        "(dates are stripped by design, not a dump bug)",
                        severity=SEVERITY_INFO,
                    )
                )
            else:
                out.append(
                    _fail(
                        "prompt.graph_timeless",
                        "prompt",
                        "graph prompt snapshot does not mention timeless relations",
                        severity=SEVERITY_WARNING,
                    )
                )
        else:
            out.append(
                _fail(
                    "prompt.graph_timeless",
                    "prompt",
                    "graph pack has no graph_v1 in the prompt bundle",
                    severity=SEVERITY_WARNING,
                )
            )
    return out


def _memory_checks(
    paths: AuditPaths,
    meta: dict[str, Any],
    memory_type: str | None,
    kind: str,
) -> list[Check]:
    idx_path = paths.memory_dir / "index.jsonl"
    if not idx_path.is_file():
        sev = SEVERITY_WARNING if kind == "thin_catalog" else SEVERITY_ERROR
        return [
            _fail(
                "memory.index",
                "memory",
                "missing memory/index.jsonl "
                "(run collect-full, or this is a thin aggregate catalog)",
                severity=sev,
            )
        ]
    rows = [
        r
        for r in load_jsonl(idx_path)
        if r.get("key_kind") in (None, "sample")
    ]
    out = [
        _ok(
            "memory.index",
            "memory",
            f"{len(rows)} sample memory row(s)",
            evidence={"n_samples": len(rows)},
        )
    ]
    empty = [r for r in rows if int(r.get("n_chars") or 0) <= 0]
    if empty:
        out.append(
            _fail(
                "memory.nonempty",
                "memory",
                f"{len(empty)} sample(s) have n_chars=0",
            )
        )
    else:
        out.append(
            _ok(
                "memory.nonempty",
                "memory",
                "all sample memory payloads have n_chars>0",
            )
        )
    if memory_type in GRAPH_METHODS and rows:
        sample_id = str(rows[0].get("sample_id") or "")
        text_path = paths.memory_dir / "by_sample" / f"{sample_id}.txt"
        if text_path.is_file():
            text = text_path.read_text(encoding="utf-8")
            graphish = "Graph relations:" in text and " -- " in text
            transcriptish = bool(SPEAKER_LINE_RE.search(text))
            if graphish and not transcriptish:
                out.append(
                    _ok(
                        "memory.graph_grammar",
                        "memory",
                        "injected {memory} is source -- rel -- target lines, "
                        "not the raw transcript",
                    )
                )
            elif graphish:
                out.append(
                    _fail(
                        "memory.graph_grammar",
                        "memory",
                        "graph payload also looks like timestamped speaker lines",
                        severity=SEVERITY_WARNING,
                    )
                )
            else:
                out.append(
                    _fail(
                        "memory.graph_grammar",
                        "memory",
                        "graph memory text is not Graph relations: triples",
                    )
                )
            n_dates = len(DATE_TOKEN_RE.findall(text))
            date_detail = (
                f"{n_dates} date-like tokens in graph {{memory}} "
                "(graph_v1 is timeless; temporal LoCoMo items "
                "cannot be answered from dates in the graph string)"
            )
            if n_dates >= 5:
                out.append(
                    _fail(
                        "memory.graph_dates",
                        "memory",
                        date_detail,
                        severity=SEVERITY_WARNING,
                        evidence={"n_date_tokens": n_dates},
                    )
                )
            else:
                out.append(
                    _ok(
                        "memory.graph_dates",
                        "memory",
                        date_detail,
                        severity=SEVERITY_INFO,
                        evidence={"n_date_tokens": n_dates},
                    )
                )
    return out


def _graph_checks(
    paths: AuditPaths, memory_type: str | None, kind: str
) -> list[Check]:
    if memory_type not in GRAPH_METHODS:
        return [
            _skip(
                "graph.dump",
                "graph",
                f"memory_type={memory_type} is not a graph writer",
            )
        ]
    gidx = load_jsonl(paths.graph_index)
    quality = load_json(paths.writer_quality)
    if not gidx:
        sev = SEVERITY_WARNING if kind == "thin_catalog" else SEVERITY_ERROR
        return [
            _fail(
                "graph.dump",
                "graph",
                "missing memory/graph/index.jsonl",
                severity=sev,
            )
        ]
    out = [_ok("graph.dump", "graph", f"{len(gidx)} graph sample dump(s)")]
    valid = [int(r.get("n_valid_edges") or 0) for r in gidx]
    if valid and min(valid) <= 0:
        out.append(
            _fail(
                "graph.edges",
                "graph",
                "at least one sample has n_valid_edges=0",
                evidence={"n_valid_edges": valid},
            )
        )
    else:
        out.append(
            _ok(
                "graph.edges",
                "graph",
                f"valid edges per sample min={min(valid)} max={max(valid)}",
                evidence={"min": min(valid), "max": max(valid), "n_samples": len(valid)},
            )
        )
    by_writer = (quality.get("by_writer") or {}) if quality else {}
    if not by_writer:
        out.append(
            _fail(
                "graph.parse_rate",
                "graph",
                "missing memory/writer/quality.json parse rates",
                severity=SEVERITY_WARNING,
            )
        )
        return out
    worst_rate = 0.0
    worst_tid = ""
    total_calls = 0
    total_fallback = 0
    for tid, rec in sorted(by_writer.items()):
        n_calls = int(rec.get("n_calls") or 0)
        n_fb = int(rec.get("n_parse_fallback") or 0)
        total_calls += n_calls
        total_fallback += n_fb
        rate = (n_fb / n_calls) if n_calls else 0.0
        if rate >= worst_rate:
            worst_rate = rate
            worst_tid = str(tid)
    if total_calls == 0:
        out.append(
            _fail("graph.parse_rate", "graph", "quality.json has zero writer calls")
        )
    elif worst_rate > PARSE_FALLBACK_MAX:
        out.append(
            _fail(
                "graph.parse_rate",
                "graph",
                f"parse fallback {worst_rate:.0%} on {worst_tid} "
                f"(threshold {PARSE_FALLBACK_MAX:.0%}). Failed JSON is replaced "
                f"with mock_extract_* triples (fallback_mock); not a live graph.",
                evidence={
                    "fallback_rate": round(worst_rate, 4),
                    "n_parse_fallback": total_fallback,
                    "n_calls": total_calls,
                    "writer_id": worst_tid,
                },
            )
        )
    else:
        out.append(
            _ok(
                "graph.parse_rate",
                "graph",
                f"parse fallback {worst_rate:.1%} (n_fallback={total_fallback}/{total_calls})",
                evidence={
                    "fallback_rate": round(worst_rate, 4),
                    "n_parse_fallback": total_fallback,
                    "n_calls": total_calls,
                },
            )
        )
    return out


def _gold_checks(paths: AuditPaths, kind: str) -> list[Check]:
    pred_path = predictions_jsonl(paths.run_dir)
    if pred_path is None:
        return [
            _skip(
                "gold.not_in_memory",
                "gold",
                "no predictions to join gold against memory_text",
            )
        ]
    leaks = 0
    n = 0
    example = None
    for row in load_jsonl(pred_path):
        gold = str(row.get("reference_answer") or "").strip()
        mem = str(row.get("memory_text") or "")
        if len(gold) < GOLD_MIN_CHARS or not mem:
            continue
        n += 1
        if gold in mem:
            leaks += 1
            if example is None:
                example = {
                    "question_id": row.get("question_id"),
                    "gold_preview": gold[:80],
                }
    if n == 0:
        return [
            _skip(
                "gold.not_in_memory",
                "gold",
                "no prediction rows with memory_text + long gold "
                "(thin catalogs omit memory_text)",
            )
        ]
    if leaks:
        return [
            _fail(
                "gold.not_in_memory",
                "gold",
                f"{leaks}/{n} rows have reference_answer inside memory_text",
                evidence={"n_leaks": leaks, "n_checked": n, "example": example},
            )
        ]
    return [
        _ok(
            "gold.not_in_memory",
            "gold",
            f"gold answers not found in memory_text (n={n})",
            evidence={"n_checked": n},
        )
    ]


def _sha256_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_run_dirs(path: str | Path) -> list[Path]:
    """Find run packs under a path. Prefers full audits over thin catalogs."""
    root = Path(path)
    if (root / "run_meta.json").is_file():
        return [root]
    found: dict[str, Path] = {}

    def _consider(directory: Path) -> None:
        if not (directory / "run_meta.json").is_file():
            return
        key = directory.name
        prev = found.get(key)
        if prev is None:
            found[key] = directory
            return
        prev_full = (prev / "memory" / "index.jsonl").is_file()
        new_full = (directory / "memory" / "index.jsonl").is_file()
        if new_full and not prev_full:
            found[key] = directory

    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                _consider(child)
        by_run = root / "aggregate" / "by_run"
        if by_run.is_dir():
            for child in sorted(by_run.iterdir()):
                _consider(child)
        collected = root / "collected" / "runs"
        if collected.is_dir():
            for child in sorted(collected.iterdir()):
                _consider(child)
    return [found[k] for k in sorted(found)]


def render_pack_markdown(report: PackReport) -> str:
    lines = [
        f"## `{report.run_id}` ({report.verdict})",
        "",
        f"- pack_kind: `{report.pack_kind}`",
        f"- memory_type: `{report.memory_type}`",
        f"- path: `{report.run_dir}`",
        "",
        "| check | layer | severity | status | detail |",
        "|---|---|---|---|---|",
    ]
    for c in report.checks:
        detail = c.detail.replace("|", "\\|")
        lines.append(
            f"| `{c.id}` | {c.layer} | {c.severity} | **{c.status}** | {detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def dump_report_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
