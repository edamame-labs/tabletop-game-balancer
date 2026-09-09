from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..spec import Params, canonical


@dataclass
class OptResult:
    best_params: Params
    best_score: float
    n_calls: int
    elapsed: float
    history: List[Tuple[int, float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class Tracker:
    """Records the best-so-far trace and prints progress."""

    def __init__(self, evaluator, verbose: bool = True, label: str = ""):
        self.ev = evaluator
        self.verbose = verbose
        self.label = label
        self.best_params: Optional[Params] = None
        self.best_score = float("-inf")
        self.history: List[Tuple[int, float]] = []
        self.t0 = time.time()
        self._offered = {}

    def refresh(self) -> None:
        """Current cached means are authoritative, including downward revisions."""
        best = self.ev.cache.best(self.ev.game, self.ev.run_type, limit=1)
        if best:
            key, self.best_score, _n, _rt = best[0]
            self.best_params = json.loads(key)
        elif self._offered:
            self.best_params, self.best_score = max(self._offered.values(),
                                                     key=lambda item: item[1])

    def offer(self, params: Params, score: float) -> bool:
        previous = self.best_score
        if not math.isfinite(score):
            self.refresh()
            return False
        self._offered[canonical(params)] = (dict(params), score)
        self.refresh()
        if self.best_score != previous:
            self.history.append((self.ev.n_calls, self.best_score))
            if self.verbose:
                print("  [%5d calls | %6.1fs] current best %.2f%s"
                      % (self.ev.n_calls, time.time() - self.t0, self.best_score,
                         (" (%s)" % self.label) if self.label else ""))
        return self.best_score > previous

    def result(self, **meta) -> OptResult:
        self.refresh()
        return OptResult(self.best_params or {}, self.best_score, self.ev.n_calls,
                         time.time() - self.t0, self.history, meta)
