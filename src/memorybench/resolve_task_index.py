"""Map CLI ``--run-index`` or Cloud Run ``CLOUD_RUN_TASK_INDEX`` to a matrix row.

Experiment code otherwise stays unaware of GCP.
"""

from __future__ import annotations

import os


def resolve_task_index(cli_index: int | None) -> int:
    if cli_index is not None:
        return int(cli_index)
    cloud_index = os.getenv("CLOUD_RUN_TASK_INDEX")
    if cloud_index is not None and str(cloud_index).strip() != "":
        return int(cloud_index)
    return 0
