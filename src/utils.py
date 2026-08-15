"""Ranking metrics used by the IJCAI experiment code."""

import torch


def cal_ranks(scores: torch.Tensor, objs: torch.Tensor):
    """Return the one-based rank of each target entity in a score matrix."""
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    ranks = (sorted_indices == objs.unsqueeze(1)).nonzero()[:, 1]
    return ranks + 1


def cal_performance(ranks):
    """Compute MRR and Hits@1/3/5/10 from one-based ranks."""
    mrr = (1.0 / ranks).sum() / len(ranks)
    h_1 = sum(ranks <= 1) * 1.0 / len(ranks)
    h_3 = sum(ranks <= 3) * 1.0 / len(ranks)
    h_5 = sum(ranks <= 5) * 1.0 / len(ranks)
    h_10 = sum(ranks <= 10) * 1.0 / len(ranks)
    return mrr, h_1, h_3, h_5, h_10
