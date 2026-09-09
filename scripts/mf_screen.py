"""Fast-screen candidates, predict medium scores, and confirm the shortlist.

Uses the same experiment storage and cached Evaluator as the main CLI.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ttbalance.cli import build_cache, refresh_entry, save_entry, storage_dir
from ttbalance.client import ModalClient
from ttbalance.evaluate import Evaluator, REJECTED
from ttbalance.multifidelity import MultiFidelityModel
from ttbalance.optimizers.surrogate import _candidates
from ttbalance.spec import load_specs


def screen(args, client, model):
    spec = load_specs()[args.game]
    cache = build_cache(args)
    try:
        # Check training readiness before spending any screening evaluations.
        if not model.fit(cache, spec, args.game):
            raise ValueError("need at least five medium configurations before screening")
        leaders = [json.loads(row[0]) for row in cache.best(args.game, "medium", limit=12, min_obs=2)]
        leaders = [p for p in leaders if not spec.validate(p)] or [spec.default()]
        candidates = _candidates(spec, leaders, random.Random(args.seed), args.pool, set(), rate=0.25)
        fast = Evaluator(client, cache, args.game, run_type="fast", workers=args.workers)
        cheap = fast.evaluate_many(candidates)
        usable = [(p, score) for p, score in zip(candidates, cheap) if score != REJECTED]
        if not usable:
            raise ValueError("screening produced no usable candidates")
        acquisition = model.acquisition(spec, [p for p, _ in usable], [s for _, s in usable])
        order = sorted(range(len(usable)), key=lambda i: -acquisition[i])[:args.top]
        picks = [usable[i][0] for i in order]
        medium = Evaluator(client, cache, args.game, run_type="medium", workers=args.workers)
        scores = medium.evaluate_many(picks, repeats=3)
        confirmed = [(p, score, len(cache.scores(args.game, p, "medium")))
                     for p, score in zip(picks, scores) if score != REJECTED]
        confirmed = [row for row in confirmed if row[2] >= 3]
        if not confirmed:
            raise ValueError("no candidate completed three medium observations")
        confirmed.sort(key=lambda row: -row[1])
        directory = os.path.join(storage_dir(args), "entries")
        refresh_entry(cache, args.game, directory)
        p, mean, n = confirmed[0]
        path = save_entry(args.game, p, mean, "medium", n, args.backend, directory)
        print("Best screened candidate %.2f (n=%d); selected entry: %s" % (mean, n, path))
        return confirmed
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", nargs="?", default="ExplodingKittens", choices=load_specs())
    parser.add_argument("pool", nargs="?", type=int, default=300)
    parser.add_argument("top", nargs="?", type=int, default=8)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cache", default=os.environ.get("TTB_CACHE", ""))
    parser.add_argument("--results-dir", default=os.environ.get("TTB_RESULTS_DIR", "results/experiments"))
    parser.add_argument("--experiment", default=os.environ.get("TTB_EXPERIMENT", "default"))
    parser.add_argument("--evaluator-version", default=os.environ.get("TTB_EVALUATOR_VERSION", "unversioned"))
    parser.set_defaults(backend="modal")
    args = parser.parse_args()
    if min(args.pool, args.top, args.workers) < 1:
        parser.error("pool, top and workers must be positive")
    try:
        screen(args, ModalClient(timeout_ms=3000000), MultiFidelityModel())
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, "error: %s\n" % exc)


if __name__ == "__main__":
    main()
