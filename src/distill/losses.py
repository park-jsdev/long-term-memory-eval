"""Simple KD loss stubs (fill in once teacher/student are wired)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_ce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha_ce: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """alpha * CE(student, labels) + (1-alpha) * T^2 * KL(student || teacher)."""
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )
    t = temperature
    s = F.log_softmax(student_logits / t, dim=-1)
    soft_t = F.softmax(teacher_logits / t, dim=-1)
    kl = F.kl_div(s, soft_t, reduction="batchmean") * (t * t)
    return alpha_ce * ce + (1.0 - alpha_ce) * kl
