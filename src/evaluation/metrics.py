from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.metrics import average_precision_score


def compute_cmc(ranks: Iterable[int], max_rank: int) -> List[float]:
    ranks = [r for r in ranks if r > 0]
    if not ranks:
        return [0.0] * max_rank
    return [float(np.mean(np.array(ranks) <= k)) for k in range(1, max_rank + 1)]


def compute_map_fair(y_true: np.ndarray, y_score: np.ndarray) -> float:
    ap = []
    for class_idx in range(y_true.shape[1]):
        if y_true[:, class_idx].sum() > 0:
            ap.append(average_precision_score(y_true[:, class_idx], y_score[:, class_idx]))
    return float(np.mean(ap)) if ap else 0.0
