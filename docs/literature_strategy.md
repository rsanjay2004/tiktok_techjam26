# Literature-Backed Strategy: Autonomous Scalable Recommender Research Agent

## Recommended project direction

Build an **Autonomous Scalable Recommender Research Agent** for KuaiRand-Pure.

The agent should not only train one recommender model. It should run a bounded R&D loop:

1. Read the current metrics and run log.
2. Choose the next experiment from a library of scalable recommender strategies.
3. Apply a small code or config change.
4. Train and evaluate on KuaiRand-Pure.
5. Record GAUC, nDCG@5, wall-clock, token usage, error recovery, and whether the change improved validation.
6. Stop when validation has converged under the challenge rule.

This is more unique than a typical team approach because the product is the **autonomous experimentation system**, not just the final recommender.

## Literature signals to use

### 1. KuaiRand makes unbiased evaluation the center of the project

KuaiRand was created because ordinary recommendation logs suffer from exposure bias. Its random-exposure logs provide missing-at-random feedback and rich user/item features, which makes it suitable for evaluating recommendation quality without simply reproducing the old platform policy.

Project implication:

- Use `log_random_4_22_to_5_08_pure.csv` for validation/test-style evaluation.
- Use standard logs and features for training signals.
- Report GAUC and nDCG@5 exactly as required.
- Include a visible note that the solution optimizes for unbiased recommendation, not just historical click imitation.

### 2. Scalable recommendation usually needs two stages

Recent industrial recommender work emphasizes scalable retrieval plus ranking, especially for long user histories. LongRetriever argues that using ultra-long behavior sequences only in ranking misses a major retrieval-stage opportunity. Its scalable idea is to produce multiple interest-specific user vectors and retrieve candidates from partitioned item repositories.

Project implication:

- For the hackathon, implement a lightweight version:
  - Build user interest profiles by category or feedback-weighted video clusters.
  - Generate multiple user contexts instead of one global user vector.
  - Retrieve candidate videos per context.
  - Rank only a small candidate set.
- This lets the project talk about scale: the expensive model never scores every user-video pair.

### 3. Multi-behavior signals matter for short-video platforms

KuaiRand includes click, like, follow, comment, forward, hate, long view, play time, profile stay time, and comment stay time. Recent heterogeneous and multi-behavior sequential recommendation literature points out that different behavior types represent different levels of intent.

Project implication:

- Avoid optimizing only `is_click`.
- Create a composite satisfaction label or multi-task setup:
  - Primary: `long_view` or `is_click`.
  - Auxiliary positives: `is_like`, `is_follow`, `is_comment`, `is_forward`.
  - Negative/risk signal: `is_hate`.
- This gives the agent richer experiment choices than simple click prediction.

### 4. Decoupled representations are a fresh scalability angle

Recent long-sequence recommendation papers argue that a single embedding table can create interference between attention/search and final representation. DARE-style decoupled embeddings are attractive because they improve long-sequence modeling and can reduce attention-side dimension for faster serving.

Project implication:

- For a lightweight implementation, approximate this with two feature spaces:
  - Retrieval features: compact hashed/category/time-decayed profile features.
  - Ranking features: richer user-video cross features and feedback aggregates.
- Pitch this as "decoupled retrieval and ranking representations" rather than a heavy Transformer.

### 5. Evaluation itself can be biased if random exposure is used carelessly

Recent debiasing evaluation work warns that evaluation on randomly exposed data can still be mishandled. The project should explicitly separate training, validation, and hidden-style evaluation logic, and should avoid overfitting to one random split.

Project implication:

- Keep validation fixed across autonomous iterations.
- Log every experiment, including failed ones.
- Prefer small, reproducible deltas over many uncontrolled changes.
- Use convergence, not cherry-picked peak score, matching the challenge criteria.

## Unique solution proposal

### Name

**AutoScaleRec: An Autonomous Agent for Scalable, Debiased Short-Video Recommendation**

### Core idea

Most teams will likely train a better model directly on KuaiRand-Pure. AutoScaleRec instead builds an autonomous research loop that learns which scalable recommender strategy works best under the challenge budget.

The agent has four modules:

1. **Requirement analyst**
   - Uses an OpenAI model to read the current run state.
   - Chooses the next experiment based on metrics, error logs, and remaining budget.

2. **Experiment planner**
   - Selects from a controlled menu:
     - Feature expansion
     - Composite satisfaction label
     - Negative sampling strategy
     - Time-decayed user profile
     - Multi-context candidate retrieval
     - Decoupled retrieval/ranking features
     - Lightweight model change

3. **Training executor**
   - Runs the selected config on KuaiRand-Pure.
   - Uses CPU-friendly models first:
     - Logistic regression
     - LightGBM / gradient boosting if allowed
     - Factorization-style features
     - Two-stage retrieval + ranker

4. **Run logger and recovery**
   - Records hypothesis, diff/config, metrics, wall-clock, token usage, and recovery.
   - If an experiment fails, rolls back to the last valid config and tries a cheaper alternative.

## Model strategy

### Baseline

Start with a CPU-fast pointwise ranker:

- Inputs:
  - user ID frequency features
  - video ID frequency features
  - user features
  - video basic/statistic features
  - time features
  - prior standard-log engagement features
- Target:
  - `long_view` first, because it better captures short-video satisfaction than raw click.
- Evaluation:
  - GAUC
  - nDCG@5

### Distinctive improvement path

1. **Composite satisfaction target**
   - Combine `long_view`, `is_like`, `is_follow`, `is_comment`, and `is_forward`.
   - Penalize `is_hate`.
   - Use this as a training target or sample weight, but still evaluate on the required benchmark label.

2. **Time-decayed user interest features**
   - Weight recent interactions more strongly.
   - Aggregate by video category, author, duration bucket, and feedback type.

3. **Multi-context candidate retrieval**
   - Build several user profiles per user:
     - long-view profile
     - like/follow profile
     - recent-watch profile
     - anti-interest profile from hate/short-view
   - Retrieve candidates per profile, merge, then rank.

4. **Decoupled retrieval/ranking features**
   - Retrieval uses cheap sparse/hash/category features.
   - Ranking uses richer cross features.
   - This directly supports the scalability story.

5. **Autonomous ablation loop**
   - Agent tries each improvement one at a time.
   - Keeps only changes that improve validation-best score.
   - Stops after convergence or budget cap.

## Why this beats typical solutions

| Typical team solution | AutoScaleRec advantage |
|---|---|
| Train one model and tune hyperparameters | Runs an autonomous experiment loop with hypotheses and recovery |
| Optimize click only | Uses multi-behavior satisfaction signals |
| Score all items directly | Uses retrieval-then-rank for scalability |
| Add deep model complexity | Starts CPU-friendly and budget-aware |
| Report final score only | Produces per-iteration logs required by judging |
| Treat KuaiRand as just another CSV | Uses the random-exposure property as the core insight |

## Benchmark against challenge pointers

| Judging pointer | How AutoScaleRec addresses it | Evidence to submit |
|---|---|---|
| Technical Execution, 35% | Implements KuaiRand-Pure pipeline, GAUC, nDCG@5, convergence rule, final checkpoint | Metrics table, reproducible command, final output schema |
| Innovation, 20% | Combines autonomous experiment planning, random-exposure debiasing, multi-behavior satisfaction, and scalable retrieval/ranking | Literature-backed README section and ablation table |
| Impact, 20% | Mimics real recommender R&D acceleration: the agent proposes, tests, and keeps/rejects changes | Iteration logs and manual-intervention count |
| Feasibility, 15% | CPU-first models, bounded search space, fixed 50-iteration cap, resource tracking | Token usage, wall-clock, iteration count |
| Robustness | Failed experiment recovery and rollback are first-class behavior | Error/recovery events in run log |
| Scalability focus | Two-stage retrieval/ranking, decoupled feature spaces, multi-context profiles | Complexity note: rank only candidate set, not full item universe |

## Three-day build plan

### Day 1

- Load KuaiRand-Pure.
- Implement train/validation split.
- Implement GAUC and nDCG@5.
- Train a simple baseline.
- Create the run-log schema.

### Day 2

- Add the autonomous experiment loop.
- Add feature families:
  - time features
  - user/video aggregate features
  - multi-behavior satisfaction features
  - time-decayed profile features
- Add rollback on failed runs.

### Day 3

- Add multi-context candidate retrieval.
- Run ablations until convergence.
- Produce final results table.
- Write final Devpost description and limitations.

## Minimum viable implementation

If time is short, build this reduced version:

1. Logistic regression or LightGBM ranker.
2. Three experiment types:
   - target/weight change
   - time-decayed aggregate features
   - negative sampling change
3. Autonomous planner that chooses one experiment per iteration.
4. Full run logs with GAUC, nDCG@5, wall-clock, and recovery events.

That is already stronger than a normal static training notebook because it directly addresses autonomy, robustness, practicality, and scalability.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Heavy deep models exceed time budget | Use CPU-friendly rankers first |
| OpenAI planner suggests risky changes | Restrict it to a safe experiment menu |
| Validation overfitting | Fixed split, convergence rule, ablation log |
| Dataset processing is slow | Cache processed features and use KuaiRand-Pure first |
| Scalability pitch feels theoretical | Implement candidate retrieval even if lightweight |

## Recommended final pitch

**AutoScaleRec is an autonomous recommender-system research agent for KuaiRand-Pure. Instead of manually tuning one model, it runs a bounded, logged R&D loop that chooses scalable recommendation experiments, trains and evaluates them, recovers from failures, and stops on convergence. Its main insight is that short-video recommendation should be optimized with random-exposure evaluation, multi-behavior satisfaction signals, and scalable retrieval-then-ranking rather than naive click prediction.**
