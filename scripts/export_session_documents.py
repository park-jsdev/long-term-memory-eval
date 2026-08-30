"""Write inspectable session-document tables under data/processed/.

Does not modify locomo10.json. Gold answers appear only in qa_joined.csv
for human audit (not in SessionBlock / session_documents.csv turn text).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import load_raw
from src.locomo_eval.preprocess.dataset_stats import write_dataset_analysis
from src.locomo_eval.preprocess.export_session_tables import export_tables

DEFAULT_DATA = ROOT / "data" / "raw" / "locomo10.json"
DEFAULT_OUT = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write histograms + naive-retrieval bars (default: true)",
    )
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path}. Run python scripts/fetch_locomo.py")
    out_dir = Path(args.out)
    samples = load_raw(data_path)
    counts = export_tables(samples, out_dir)
    (out_dir / "export_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Wrote tables under {out_dir}")
    if args.plots:
        plot_dir = write_dataset_analysis(samples, out_dir)
        print(f"Wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
