"""Noise-aware, cached, parallel evaluation of game configurations.

`fast` runs are only 36 matchups, so a single score is a noisy estimate of the
configuration's true quality. Every optimiser here works against `Evaluator`,
which averages repeated observations and never pays twice for the same work.
"""
from __future__ import annotations

import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .cache import Cache
from .client import ApiError, BaseClient, RunRejected
from .spec import Params, canonical

REJECTED = float("-inf")


class BudgetExhausted(RuntimeError):
    pass


class CallBudget:
    """One thread-safe evaluation budget, optionally shared across games."""

    def __init__(self, limit: Optional[int] = None):
        if limit is not None and limit < 0:
            raise ValueError("budget must be nonnegative")
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> float:
        with self._lock:
            return math.inf if self.limit is None else self.limit - self.used

    def take(self, n: int = 1) -> None:
        with self._lock:
            if self.limit is not None and self.used + n > self.limit:
                raise BudgetExhausted("API budget of %d evaluations exhausted" % self.limit)
            self.used += n


class Evaluator:
    def __init__(self, client: BaseClient, cache: Cache, game: str,
                 run_type: str = "fast", repeats: int = 1, workers: int = 4,
                 budget: Optional[Union[int, CallBudget]] = None, verbose: bool = True):
        if repeats < 1 or workers < 1:
            raise ValueError("repeats and workers must be positive")
        cache.bind(client.cache_identity)
        self.client = client
        self.cache = cache
        self.game = game
        self.run_type = run_type
        self.repeats = repeats
        self.workers = workers
        self._budget = budget if isinstance(budget, CallBudget) else CallBudget(budget)
        self.budget = self._budget.limit
        self.verbose = verbose
        self.n_calls = 0
        self.n_cached = 0
        self.n_failed = 0
        self.n_rejected = 0
        self.exhausted = False
        self._lock = threading.Lock()
        self._rejected = set()

    # -- budget ---------------------------------------------------------
    def _take(self, n: int = 1) -> None:
        try:
            self._budget.take(n)
        except BudgetExhausted:
            self.exhausted = True
            raise
        with self._lock:
            self.n_calls += n

    def check(self) -> None:
        """Raise if the budget ran out. `evaluate_many` absorbs exhaustion so a
        partly-finished batch is still usable; optimisers call this once they
        have consumed the batch, which is what actually stops their loops."""
        if self.exhausted:
            raise BudgetExhausted("API budget of %d calls exhausted" % self.budget)

    @property
    def remaining(self) -> float:
        return self._budget.remaining

    # -- core -----------------------------------------------------------
    def observe(self, params: Params, run_type: Optional[str] = None,
                repeats: Optional[int] = None) -> List[float]:
        """Ensure at least `repeats` observations exist; return all of them."""
        rt = run_type or self.run_type
        want = self.repeats if repeats is None else repeats
        if want < 1:
            raise ValueError("repeats must be positive")
        with self.cache.evaluation_lock(self.game, params, rt):
            return self._observe_locked(params, rt, want)

    def _observe_locked(self, params: Params, rt: str, want: int) -> List[float]:
        key = (canonical(params), rt)
        if key in self._rejected:
            return []
        have = self.cache.scores(self.game, params, rt)
        with self._lock:
            self.n_cached += min(len(have), want)
        while len(have) < want:
            self._take()
            try:
                score = self.client.score(self.game, params, rt)
            except RunRejected as exc:
                self._rejected.add(key)
                with self._lock:
                    self.n_rejected += 1
                if self.verbose:
                    print("  ! rejected: %s" % exc, file=sys.stderr)
                return []
            except ApiError as exc:
                with self._lock:
                    self.n_failed += 1
                print("  ! backend failure (%s/%s): %s" %
                      (self.game, rt, exc), file=sys.stderr)
                raise
            self.cache.add(self.game, params, rt, score)
            have.append(score)
        return have

    def evaluate(self, params: Params, run_type: Optional[str] = None,
                 repeats: Optional[int] = None) -> float:
        scores = self.observe(params, run_type, repeats)
        return REJECTED if not scores else sum(scores) / len(scores)

    def evaluate_many(self, batch: Sequence[Params], run_type: Optional[str] = None,
                      repeats: Optional[int] = None,
                      on_result: Optional[Callable[[Params, float], None]] = None
                      ) -> List[float]:
        """Evaluate a batch in parallel. Budget exhaustion truncates the batch
        rather than losing the work already done."""
        groups = {}
        for i, p in enumerate(batch):
            groups.setdefault(canonical(p), []).append(i)
        unique = [(indices, batch[indices[0]]) for indices in groups.values()]
        out: List[float] = [REJECTED] * len(batch)
        failures = []

        def job(item) -> None:
            indices, p = item
            try:
                score = self.evaluate(p, run_type, repeats)
            except BudgetExhausted:
                # Keep paid observations even if the repeat target was cut short.
                obs = self.cache.scores(self.game, p, run_type or self.run_type)
                score = sum(obs) / len(obs) if obs else REJECTED
            except ApiError as exc:
                with self._lock:
                    failures.append(str(exc))
                score = REJECTED
            for i in indices:
                out[i] = score
                if on_result:
                    on_result(batch[i], score)

        if self.workers <= 1:
            for item in unique:
                job(item)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(job, unique))
        if failures and all(s == REJECTED for s in out):
            raise ApiError("batch produced no usable scores (%d backend failures): %s" %
                           (len(failures), failures[0]))
        return out

    # -- noise handling -------------------------------------------------
    def race(self, candidates: Sequence[Params], keep: int = 1,
             rounds: Sequence[int] = (1, 3, 7), run_type: Optional[str] = None
             ) -> List[Tuple[Params, float, int]]:
        """Successive halving over repeated evaluations.

        Cheap single looks at everything, then progressively more repeats on the
        survivors. This is what stops the search from crowning a configuration
        that merely got a lucky 36-matchup draw.
        """
        if keep < 1 or not rounds or any(n < 1 for n in rounds):
            raise ValueError("race requires positive keep and repeat targets")
        if list(rounds) != sorted(set(rounds)):
            raise ValueError("race repeat targets must strictly increase")
        pool = list({canonical(p): p for p in candidates}.values())
        for r, reps in enumerate(rounds):
            if not pool:
                break
            means = self.evaluate_many(pool, run_type=run_type, repeats=reps)
            if self.exhausted and all(m == REJECTED for m in means):
                break
            ranked = sorted(zip(pool, means), key=lambda t: -t[1])
            ranked = [(p, m) for p, m in ranked if m != REJECTED]
            if r < len(rounds) - 1:
                width = max(keep, len(ranked) // 2)
                pool = [p for p, _ in ranked[:width]]
                if self.verbose:
                    print("  race round %d (reps=%d): %d -> %d survivors, "
                          "best %.2f" % (r + 1, reps, len(means), len(pool),
                                         ranked[0][1] if ranked else float("nan")))
            else:
                pool = [p for p, _ in ranked]
        final = []
        for p in pool[:keep]:
            obs = self.cache.scores(self.game, p, run_type or self.run_type)
            if obs:
                final.append((p, sum(obs) / len(obs), len(obs)))
        return final

    def summary(self) -> str:
        return ("%s/%s: %d evaluation calls, %d cache hits, %d backend failures, "
                "%d rejected configurations" % (self.game, self.run_type, self.n_calls,
                self.n_cached, self.n_failed, self.n_rejected))
