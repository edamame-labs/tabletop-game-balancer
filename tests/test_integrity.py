"""Offline regressions for result provenance and expensive evaluation orchestration."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ttbalance import cli
from ttbalance.cache import Cache
from ttbalance.client import ApiError, BaseClient, LocalClient, LocalPoolClient, make_client
from ttbalance.evaluate import BudgetExhausted, CallBudget, Evaluator
from ttbalance.mock import MockClient
from ttbalance.optimizers.base import Tracker
from ttbalance.spec import canonical, load_specs

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_specs()["CantStop"]


class CountingClient(BaseClient):
    name = "local"
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()
    def score(self, game, params, run_type="fast"):
        with self.lock:
            self.calls += 1
        return 900.0


class TestIntegrity(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name
        self.client = CountingClient()
        self.cache = Cache(os.path.join(self.directory, "cache.sqlite"))
        self.addCleanup(self.cache.close)
        self.ev = Evaluator(self.client, self.cache, "CantStop", verbose=False)
        self.p = SPEC.default()
        self.q = SPEC.mutate(self.p, random.Random(0), rate=1.0)

    def test_real_evaluator_refuses_mock_observations(self):
        cache = Cache(":memory:")
        self.addCleanup(cache.close)
        Evaluator(MockClient(1), cache, "CantStop", verbose=False).evaluate(self.p)
        with self.assertRaisesRegex(ValueError, "different evaluator"):
            Evaluator(self.client, cache, "CantStop", verbose=False)
        self.assertEqual(self.client.calls, 0)

    def test_provenance_survives_reopening_and_versions_do_not_mix(self):
        self.ev.evaluate(self.p)
        other = Cache(self.cache.path, context={"evaluator_version": "changed"})
        self.addCleanup(other.close)
        with self.assertRaisesRegex(ValueError, "different evaluator"):
            other.bind(self.client.cache_identity)
        self.assertEqual(other.scores("CantStop", self.p, "fast"), [900.0])

    def test_unlabelled_legacy_data_is_preserved_but_not_reused(self):
        legacy = Cache(":memory:")
        self.addCleanup(legacy.close)
        legacy.add("CantStop", self.p, "fast", 999)
        with self.assertRaisesRegex(ValueError, "without evaluator provenance"):
            Evaluator(self.client, legacy, "CantStop", verbose=False)
        self.assertEqual(legacy.scores("CantStop", self.p, "fast"), [999])

    def test_local_pool_and_single_client_share_evaluator_identity(self):
        self.assertEqual(LocalClient().cache_identity,
                         LocalPoolClient(["http://localhost:3009/api/"]).cache_identity)

    def test_mock_seed_is_part_of_provenance(self):
        self.assertNotEqual(MockClient(1).cache_identity, MockClient(2).cache_identity)

    def test_tracker_demotes_lucky_winner_after_confirmation(self):
        tr = Tracker(self.ev, verbose=False)
        self.cache.add("CantStop", self.p, "fast", 1000)
        tr.offer(self.p, 1000)
        self.cache.add("CantStop", self.q, "fast", 900)
        tr.offer(self.q, 900)
        self.cache.add("CantStop", self.p, "fast", 700)
        # Refreshes need not have been offered by the optimizer.
        result = tr.result()
        self.assertEqual(result.best_params, self.q)
        self.assertEqual(result.best_score, 900)

    def test_tracker_offer_refreshes_downward_and_changes_incumbent(self):
        tr = Tracker(self.ev, verbose=False)
        tr.offer(self.p, 1000)
        tr.offer(self.q, 900)
        tr.offer(self.p, 850)
        self.assertEqual(tr.best_params, self.q)
        self.assertEqual(tr.best_score, 900)

    def test_more_observations_do_not_promote_a_worse_different_entry(self):
        cli.save_entry("CantStop", self.p, 950, "medium", 3, "local", self.directory)
        cli.save_entry("CantStop", self.q, 800, "medium", 4, "local", self.directory)
        self.assertEqual(self.read_entry()["params"], self.p)
        self.assertEqual(self.read_entry()["score"], 950)

    def test_same_entry_gets_its_revised_mean(self):
        cli.save_entry("CantStop", self.p, 950, "medium", 3, "local", self.directory)
        cli.save_entry("CantStop", self.p, 800, "medium", 4, "local", self.directory)
        self.assertEqual(self.read_entry()["score"], 800)

    def test_better_and_equally_confirmed_entry_replaces_incumbent(self):
        cli.save_entry("CantStop", self.p, 800, "medium", 3, "local", self.directory)
        cli.save_entry("CantStop", self.q, 900, "medium", 3, "local", self.directory)
        self.assertEqual(self.read_entry()["params"], self.q)

    def test_entry_rejects_backend_mixing_and_nonfinite_scores(self):
        cli.save_entry("CantStop", self.p, 900, "medium", 3, "local", self.directory)
        with self.assertRaisesRegex(ValueError, "different evaluator"):
            cli.save_entry("CantStop", self.q, 999, "full", 9, "mock", self.directory)
        with self.assertRaises(ValueError):
            cli.save_entry("CantStop", self.q, float("nan"), "medium", 3, "local", self.directory)

    def read_entry(self):
        return json.loads(Path(cli.entry_path("CantStop", self.directory)).read_text())

    def test_single_pool_honors_port_and_explicit_url(self):
        self.assertEqual(make_client("local", pool=1, first_port=3009).base_url,
                         "http://localhost:3009/api/")
        self.assertEqual(make_client("local", pool=1, first_port=3009,
                                     local_url="http://custom:4000/api").base_url,
                         "http://custom:4000/api/")

    def test_duplicate_batch_reuses_work_and_preserves_order_callbacks(self):
        callbacks = []
        result = self.ev.evaluate_many([self.p, self.q, self.p], repeats=3,
                                       on_result=lambda p, s: callbacks.append((canonical(p), s)))
        self.assertEqual(result, [900, 900, 900])
        self.assertEqual(self.client.calls, 6)
        self.assertEqual(len(callbacks), 3)
        self.assertEqual(len(self.cache.scores("CantStop", self.p, "fast")), 3)

    def test_concurrent_evaluators_share_inflight_configuration(self):
        entered, release = threading.Event(), threading.Event()
        def blocking(*args):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test worker was not released")
            return 900
        second = Evaluator(self.client, self.cache, "CantStop", verbose=False)
        with patch.object(self.client, "score", side_effect=blocking) as score:
            with ThreadPoolExecutor(2) as workers:
                first = workers.submit(self.ev.evaluate, self.p)
                self.assertTrue(entered.wait(2))
                other = workers.submit(second.evaluate, self.p)
                release.set()
                self.assertEqual([first.result(), other.result()], [900, 900])
            self.assertEqual(score.call_count, 1)

    def test_shared_budget_is_atomic_across_games(self):
        budget = CallBudget(1)
        other = Evaluator(self.client, self.cache, "Dominion", budget=budget, verbose=False)
        first = Evaluator(self.client, self.cache, "CantStop", budget=budget, verbose=False)
        first.evaluate(self.p)
        with self.assertRaises(BudgetExhausted):
            other.evaluate(load_specs()["Dominion"].default())
        self.assertEqual(self.client.calls, 1)

    def test_partial_repeat_budget_retains_paid_observation(self):
        ev = Evaluator(self.client, self.cache, "CantStop", budget=1, verbose=False)
        self.assertEqual(ev.evaluate_many([self.p], repeats=3), [900])
        self.assertTrue(ev.exhausted)

    def test_all_backend_failures_are_visible_and_abort_batch(self):
        stderr = io.StringIO()
        with patch.object(self.client, "score", side_effect=ApiError("offline")), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(ApiError, "no usable scores"):
                self.ev.evaluate_many([self.p, self.q])
        self.assertEqual(self.ev.n_failed, 2)
        self.assertIn("offline", stderr.getvalue())
        self.assertIn("2 backend failures", self.ev.summary())

    def test_partial_backend_failure_keeps_successful_results(self):
        def score(game, p, rt):
            if p == self.p:
                raise ApiError("offline")
            return 910
        with patch.object(self.client, "score", side_effect=score), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.ev.evaluate_many([self.p, self.q]), [float("-inf"), 910])

    def test_programming_error_is_not_disguised_as_rejection(self):
        with patch.object(self.client, "score", side_effect=TypeError("bug")):
            with self.assertRaisesRegex(TypeError, "bug"):
                self.ev.evaluate_many([self.p])

    def run_cli(self, argv):
        with patch.object(cli, "build_client", return_value=self.client), contextlib.redirect_stdout(io.StringIO()):
            cli.main(argv + ["--backend", "local", "--results-dir", self.directory, "--quiet"])

    def test_search_budget_applies_across_all_games(self):
        self.run_cli(["search", "--game", "CantStop", "ExplodingKittens", "--budget", "1",
                      "--optimizer", "random", "--iterations", "1", "--seed-from-cache", "0"])
        self.assertEqual(self.client.calls, 1)

    def test_verify_budget_is_shared_and_incomplete_confirmation_not_saved(self):
        self.run_cli(["search", "--game", "CantStop", "ExplodingKittens", "--budget", "2",
                      "--optimizer", "random", "--iterations", "1", "--seed-from-cache", "0"])
        self.client.calls = 0
        self.run_cli(["verify", "--game", "CantStop", "ExplodingKittens", "--budget", "1",
                      "--run-type", "medium", "--rounds", "1", "3"])
        self.assertEqual(self.client.calls, 1)
        for game in ["CantStop", "ExplodingKittens"]:
            path = Path(self.directory) / "default/local/unversioned/entries" / (game + ".json")
            self.assertEqual(json.loads(path.read_text())["run_type"], "fast")

    def test_cli_reports_backend_outage_with_failure_exit(self):
        with patch.object(self.client, "score", side_effect=ApiError("offline")), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.run_cli(["search", "--game", "CantStop", "--optimizer", "random",
                              "--iterations", "1", "--seed-from-cache", "0"])
        self.assertEqual(error.exception.code, 1)

    def test_cli_default_paths_isolate_backend_seed_version_and_experiment(self):
        def parse(extra):
            with patch.object(cli, "cmd_search") as command:
                cli.main(["search"] + extra)
            return command.call_args.args[0]
        variants = [[], ["--backend", "local"], ["--seed", "2"],
                    ["--evaluator-version", "v2"], ["--experiment", "another"]]
        self.assertEqual(len({cli.storage_dir(parse(v)) for v in variants}), len(variants))
        args = parse(["--backend", "local", "--local-pool", "1", "--first-port", "3009"])
        self.assertEqual(cli.build_client(args).base_url, "http://localhost:3009/api/")


class TestReproductionScripts(unittest.TestCase):
    def commands(self, script, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TTB_")}
        env.update(DRY_RUN="1", PYTHON=sys.executable, RUN_TYPE="medium")
        env.update(overrides)
        result = subprocess.run(["bash", str(ROOT / "scripts" / script)], cwd=ROOT,
                                env=env, text=True, capture_output=True, check=True)
        return [shlex.split(line) for line in result.stdout.splitlines() if "-m ttbalance" in line]

    def test_search_and_confirmation_use_medium_and_disjoint_ports(self):
        searches = self.commands("search_all.sh")
        self.assertEqual(len(searches), 4)
        ports = []
        for command in searches:
            self.assertEqual(command[command.index("--run-type") + 1], "medium")
            first = int(command[command.index("--first-port") + 1])
            size = int(command[command.index("--local-pool") + 1])
            ports.extend(range(first, first + size))
        self.assertEqual(sorted(ports), list(range(3000, 3010)))
        confirmations = self.commands("verify_all.sh")
        self.assertEqual(len(confirmations), 4)
        for command in confirmations:
            self.assertEqual(command[command.index("--run-type") + 1], "medium")
            self.assertEqual(command[command.index("--from-run-type") + 1], "medium")

    def test_fidelity_override_reaches_both_stages(self):
        for script in ["search_all.sh", "verify_all.sh"]:
            for command in self.commands(script, RUN_TYPE="fast", FROM_RUN_TYPE="fast"):
                self.assertEqual(command[command.index("--run-type") + 1], "fast")


class TestScreening(unittest.TestCase):
    def setUp(self):
        import importlib.util
        from types import SimpleNamespace
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        spec = importlib.util.spec_from_file_location("mf_screen", ROOT / "scripts/mf_screen.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.args = SimpleNamespace(game="CantStop", pool=3, top=2, workers=1, seed=11,
                                    backend="local", cache="", results_dir=temporary.name,
                                    experiment="screen", evaluator_version="v1")
        self.client = CountingClient()
        class Model:
            def fit(self, cache, spec, game):
                return len(cache.best(game, "medium")) >= 5
            def acquisition(self, spec, candidates, cheap):
                return cheap
        self.model = Model()

    def seed_training(self):
        cache = cli.build_cache(self.args)
        rng = random.Random(7)
        for _ in range(5):
            cache.add("CantStop", SPEC.sample(rng), "medium", 800)
        cache.close()

    def screen(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.module.screen(self.args, self.client, self.model)

    def test_insufficient_training_spends_no_evaluations(self):
        with self.assertRaisesRegex(ValueError, "five medium"):
            self.screen()
        self.assertEqual(self.client.calls, 0)

    def test_zero_scores_are_valid_and_second_screen_reuses_cache(self):
        self.seed_training()
        rng = random.Random(42)
        candidates = [SPEC.sample(rng) for _ in range(3)]
        # Compare repeated work on the same pool; new leader observations can
        # legitimately change the candidates proposed by a later screening run.
        with patch.object(self.module, "_candidates", return_value=candidates), patch.object(self.client, "score", return_value=0) as score:
            results = self.screen()
            count = score.call_count
            self.assertEqual(count, 3 + 2 * 3)
            self.assertTrue(all(mean == 0 and n == 3 for _, mean, n in results))
            self.screen()
            self.assertEqual(score.call_count, count)

    def test_failed_confirmation_does_not_create_an_entry(self):
        self.seed_training()
        def score(game, params, run_type):
            if run_type == "medium":
                raise ApiError("confirmation offline")
            return 900
        with patch.object(self.client, "score", side_effect=score), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(ApiError):
                self.screen()
        self.assertFalse((Path(cli.storage_dir(self.args)) / "entries/CantStop.json").exists())
