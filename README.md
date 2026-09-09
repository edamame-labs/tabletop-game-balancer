# Autoresearch When Every Experiment Is Expensive

Cheap experiments make autoresearch practical. The hard part is deciding which
results to trust.

We ran into this while tuning rules for four tabletop games. After roughly
9,000 evaluations, we finished **2nd in the 2026 Tabletop Games Balancing
Competition**, with **3652.5 / 4000 points**
([official results](https://balance-competition.tabletopgames.ai/)).

We tried several search methods, including Nori, a tabular foundation model.
It helped us find useful candidates. Our bigger lesson was how much the result
depended on measuring those candidates well.

This is a wrap-up of what worked, what fooled us, and what we would try next.

## The expensive part was getting an answer

The setup was a black-box optimization problem: choose a game's rules, run
simulations, and get a score. We tuned Dominion, Exploding Kittens, 7 Wonders,
and Can't Stop. Each contributed up to 1000 points, based on how closely fixed
AI agents' win rates matched a target. Some targets favored stronger agents;
“balanced” did not mean every matchup should be 50/50.

Proposing a new configuration was easy. Evaluating it could take hours.

| Evaluation | Simulated games | Typical time in our runs |
|---|---:|---|
| `fast` | 36 | 2–15 minutes |
| `medium` | 360 | 15–40 minutes |
| `full` | 3,600 | Hours |

The natural move was to research on smaller samples and save full evaluations
for the most promising candidates. Training experiments often follow the same
pattern: use less data or fewer steps, then scale up what looks good. Our
measurements here come from game simulations, but that broader research problem
motivated the project.

This is where autoresearch gets interesting. An automated loop needs to decide
**what to try, how much to spend, and when there is enough evidence to keep a
change**. A shorter loop gives us more experiments; we still have to establish
that their improvements survive the larger evaluation.

## Cheap wins can disappear

Early on, we assembled a bundle from each game's best individual observation.
It appeared to score about **3781**. Remeasuring it brought that estimate down
to about **3485**.

We had selected winners from noisy measurements. Searching more candidates gave
us more chances to find a good configuration—and more chances to find a lucky
score. We started comparing repeated means and spending extra evaluations on
the candidates that survived an initial screen.

Repeats help with noise. Transfer is a separate question: a smaller evaluation
can rank candidates differently from the full one, even when its measurements
look convincing.

One submission made this painfully clear:

| Change | Estimated gain at `medium` | Observed gain at `full` |
|---|---:|---:|
| Update three games: v6 → v7 | +44.9 | +1.6 |
| Revert only 7 Wonders from v7 | −14.8 | +30.6 |

The revert looked worse in our research setting and better on the leaderboard.
We also replaced Dominion with a candidate that scored higher at `fast`; that
submission lost **32.4 full-evaluation points** against our best bundle.

Those results pushed us toward changing one game at a time and checking selected
changes at full fidelity. A single full run still has noise, but a smaller change
makes the comparison easier to interpret.

Our working rule became: **use cheap evaluations to generate leads, repeat
measurements to challenge apparent winners, and reserve budget to check transfer.**
A better searcher helps choose experiments; it cannot make an unreliable proxy
reliable by itself.

## A tabular foundation model as the searcher

Our experiment history was already a table: game parameters in the columns,
measured scores as labels. That made a tabular foundation model a natural
surrogate—a model that predicts an expensive evaluation before we run it.

We explored random search, hill climbing, an evolutionary algorithm, PBIL
(which learns a distribution over parameter choices), and
[Nori](https://github.com/Synthefy/synthefy-nori). Nori uses labelled rows as
context to predict new ones, without task-specific gradient training. We found
that interface convenient for a search history that kept growing.

The main Nori loop was straightforward:

1. Give it observed configurations and their mean `medium` scores.
2. Generate about 2,000 candidates through random sampling and mutation.
3. Pick about 10 with promising predictions, an exploration bonus, and some
   diversity within the batch.
4. Evaluate them, recheck the leaders, and repeat.

The exploration bonus used the spread of Nori's predicted quantiles. We treated
it as a heuristic; we did not validate that spread as a calibrated measure of
uncertainty. The implementation is in
[`surrogate.py`](src/ttbalance/optimizers/surrogate.py).

Our campaign notes attribute three of the four final configurations to direct
medium search with this loop. It contributed useful candidates. We did not run
equal-budget comparisons against the other searchers, so the second-place
finish cannot tell us how much Nori improved the outcome. Our judgment is that
measurement, rechecking, and transfer decisions mattered more.

### Could the cheap score still be useful?

Instead of trusting `fast` to rank candidates directly, we also tried using it
as one more input to the model:

```text
parameters + fast score → predicted medium score
```

In a small historical leave-one-out comparison on 18 Exploding Kittens
configurations, adding the fast score reduced prediction error from **28.21 to
25.97 MAE**. A mean baseline scored 35.10.

That was encouraging: a weak proxy may still add information. It was a small
prediction experiment, though, and we did not establish a gain in search
performance or full scores. The code lives in
[`multifidelity.py`](src/ttbalance/multifidelity.py) and
[`mf_screen.py`](scripts/mf_screen.py).

## What we take from this

We like tabular foundation models as an entry point for expensive black-box
optimization. Configurations fit naturally into rows, observations are costly,
and a pretrained model offers a convenient way to propose the next batch.
We would still compare it with simpler searchers and other surrogates before
committing a large evaluation budget.

For autoresearch, we think **evaluation deserves as much design effort as
candidate generation**. [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
shows the appeal of a short automated experiment loop. When the real objective
uses a larger budget, deciding which small-scale insights transfer becomes
part of the research itself.

Our next experiment would compare searchers from the same starting data under
equal evaluation cost, repeat across seeds, and test their selected candidates
on fresh full runs. That would give us stronger evidence about the value of the
model and the cheap-score feature.

For now, this repository shares a working search system and a second-place
case study. The code automates configuration search and repeated evaluation;
strategic decisions and final submissions involved human judgment.

*Evidence note: the winning bundle and all 14 submission scores are archived
below. The roughly 9,000 evaluations, remeasurement examples, and prediction
experiment are from campaign notes; the complete observation database and
leave-one-out script are not included.*

## Try it

Use Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
./scripts/reproduce.sh check
```

This runs the 72 offline tests and a mock optimization loop. It checks the
plumbing without an evaluator or a model download. Nori tests use a fake
predictor; they do not reproduce the model's historical performance.

The [winning submission](results/winning_submission.json) contains the final
rules. The implementation is under [`src/ttbalance/`](src/ttbalance/), and
[`scripts/`](scripts/) contains the search and confirmation workflows.

<details>
<summary><strong>Run a real search</strong></summary>

These steps need Docker and take hours. They run a new search, so they will not
reconstruct the original competition trajectory.

```bash
./scripts/reproduce.sh evaluator   # start 10 local evaluator containers
./scripts/reproduce.sh search      # launch one medium-search loop per game
# Stop the search loops before confirmation uses their container pool.
./scripts/reproduce.sh confirm     # race the leaders to 7 measurements
./scripts/reproduce.sh diagnose    # inspect variation in collected scores
./scripts/reproduce.sh bundle      # print the submission JSON
```

Search and confirmation default to `medium`. Set `RUN_TYPE=fast` to change that,
`PASSES=1` to bound each search loop, or use `DRY_RUN=1 ./scripts/search_all.sh`
and `DRY_RUN=1 ./scripts/verify_all.sh` to inspect commands first.

Optional dependencies:

- `requirements-nori.txt`: local Nori inference; downloads the public checkpoint
  on first use. See [upstream setup](https://github.com/Synthefy/synthefy-nori#install).
- `requirements-modal.txt`: Modal backend; requires an account and
  `modal deploy modal_localapi.py`, then `--backend modal`.
- `scripts/search_modal_medium.sh` expects both sets installed in `.venv-nori`.

The base CLI only needs `requirements.txt`.

</details>

<details>
<summary><strong>Experiment storage and evaluation budgets</strong></summary>

Generated observations, entries, and optimizer state live under:

```text
results/experiments/<experiment>/<backend>/<evaluator-version>/
  cache.sqlite
  entries/<game>.json
  state/<run-type>/pbil_<game>.json
```

Use `--experiment` and `--evaluator-version` consistently across commands, or set
`TTB_EXPERIMENT` and `TTB_EVALUATOR_VERSION`. The version defaults to `unversioned`;
set it to a stable image or revision label when the evaluator changes.
`--results-dir` / `TTB_RESULTS_DIR` relocates generated files. Mock paths include
the seed because it changes the objective.

An explicit `--cache` / `TTB_CACHE` must match its stored provenance. Old databases
without provenance and legacy `results/entries` or `results/state` files are
preserved but not automatically imported. Start a fresh experiment and keep the
old files for inspection.

`--budget` is shared across games within search, verify, and probe. It counts
calls to `score`, including failures; internal transport retries are not counted
separately. `bench` gives each optimizer/game case its own budget. Incomplete
confirmations do not replace an entry.

Duplicate work is coordinated within a shared `Cache` object in one process.
Separate processes need disjoint container pools. Backend errors are reported;
a batch without usable scores fails, and programming errors propagate.

</details>

<details>
<summary><strong>Competition and evaluator reference</strong></summary>

The 2026 competition closed on 1 September and announced winners on 3 September.
The [official games page](https://balance-competition.tabletopgames.ai/games)
provides the scoring formulas and matchup targets. Fixed agents range from
Random to tuned MCTS. Three games use two players; 7 Wonders uses four.

[`config/valid_params.json`](config/valid_params.json) defines accepted rules:

| Game identifier | Parameters | Special constraints |
|---|---:|---|
| `Dominion` | 10 | `CARDS`: exactly 10 of 26; use `PILES_EXHAUSTED_FOR_GAME_END` |
| `ExplodingKittens` | 14 | One boolean; the rest are integers |
| `Wonders7` | 29 | `wonders`: choose 4–7 |
| `CantStop` | 13 | Column maxima, `COLUMNS_TO_WIN`, and `MARKERS` |

[`client.py`](src/ttbalance/client.py) supports local, pooled local, hosted, and
Modal evaluators. A [local evaluator](https://balance-competition.tabletopgames.ai/localsetup)
needs no API key:

```bash
docker run --rm -p 3000:3000 longhousedev/localapi
```

Send `{game, params, run_type, timeout?}` to
`POST http://localhost:3000/api/run_game`; `timeout` is in milliseconds.
Use one request per container. [`localapi.sh`](scripts/localapi.sh) manages a
local pool, [`cloud_localapi.sh`](scripts/cloud_localapi.sh) sets up a VM pool,
and [`modal_localapi.py`](modal_localapi.py) runs evaluations on Modal.

The [hosted API](https://balance-competition.tabletopgames.ai/documentation)
uses `TTB_API_KEY` and `/submit_run`, `/query_run`, and `/retrieve_result`.
Availability after the competition depends on the organisers. Competition
entries were submitted through the website; the evaluation API did not post
them to the leaderboard.

</details>

<details>
<summary><strong>All 14 submissions and final measurements</strong></summary>

Ordered by full score, not submission time. Each change refers to its own
baseline, which may differ from the preceding row. Decimal scores come from
our records; the public leaderboard rounds them.

| Entry | Full score | Recorded change |
|---|---:|---|
| `probe-w7-lo` | **3652.5** | 7 Wonders: a low-medium candidate |
| `probe-w7-hi` | 3648.3 | 7 Wonders: the highest-medium candidate |
| `probe-ek-895` | 3642.4 | Exploding Kittens: candidate then estimated at 895.4, n=7 |
| `probe-w7-revert` | 3637.8 | Revert 7 Wonders from v7 to its v6 configuration |
| `probe-cs-revert` | 3636.3 | Revert Can't Stop; the newer configuration was retained |
| `sub-B-dom-w7div` | 3627.7 | Dominion swap and a different 7 Wonders configuration |
| `sub-A-dominion` | 3620.1 | Higher-fast Dominion candidate; 32.4 below `probe-w7-lo` |
| `medium-direct-v7` | 3607.2 | Three games changed together |
| `medium-direct-v6` | 3605.6 | Can't Stop changed |
| `medium-direct-v5` | 3562.3 | All four games changed, one using a single measurement |
| `medium-direct-v4` | 3553.3 | Exploding Kittens changed |
| `medium-direct-v3` | 3513.7 | Exploding Kittens changed |
| `medium-confirmed-v2` | 3493.1 | Bundle confirmed at three or more repeats |
| `confirmed-n7-v1` | 3490.0 | First recorded submission |

The final bundle's recorded medium measurements were:

| Game | Medium mean | Observations |
|---|---:|---:|
| Dominion | 956.13 | 3 |
| Exploding Kittens | 889.22 | 12 |
| 7 Wonders | 898.02 | 7 |
| Can't Stop | 910.92 | 7 |

The official 3652.5 is a separately measured full total. The historical
projections in [`transfer.py`](src/ttbalance/transfer.py) and
[`calibrate.py`](src/ttbalance/calibrate.py) are exploratory summaries, not
calibrated confidence bounds.

</details>

Thanks to the [TAG framework](https://github.com/GAIGResearch/TabletopGames), the
competition organisers, and [Synthefy](https://github.com/Synthefy/synthefy-nori)
for making these experiments possible. For related work, see
[multi-fidelity optimization with unreliable information sources](https://proceedings.mlr.press/v206/mikkola23a.html).
