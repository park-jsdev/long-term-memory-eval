# Data

## LoCoMo (eval / optional train)

- Source: https://github.com/snap-research/locomo
- Pinned commit: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
- File: `data/locomo10.json`
- License: **CC BY-NC 4.0** (research / non-commercial only)

Fetch:

```bash
python scripts/fetch_locomo.py
```

This writes `data/raw/locomo10.json` and prints a SHA256 for your notes.

Do not commit the raw JSON unless you intentionally accept redistribution under CC BY-NC attribution rules.
