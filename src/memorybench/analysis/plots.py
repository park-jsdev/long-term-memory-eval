"""Compat re-export. Plot engines live in ``scripts.analysis.campaign_plots``."""

from scripts.analysis.campaign_plots import (  # noqa: F401
    _place_legend_outside,
    _pyplot,
    unbounded_metric,
    write_bar,
    write_grouped_bar,
    write_metrics_grouped_bar,
)
