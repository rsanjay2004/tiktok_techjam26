# Critical Review of the Current Project

## Updated honest status (September 2026)

The project now has a real KuaiRand-Pure benchmark path and a multi-provider requirement-analysis layer. The dashboard planning loop is intentionally a scripted planning demonstration; it must not be confused with measured benchmark experiments.

The latest full-data run evaluated 1,141,112 training rows, 124,909 validation rows, and 170,588 test rows. FM remained the selected model with test primary `0.5953`, GAUC `0.6621`, and nDCG@5 `0.5286`. All current challengers were rejected because they did not beat FM on validation. The run generated validated submission CSVs, `outputs/selected_fm_checkpoint.npz`, `outputs/final_metrics.json`, and a 12-entry `runs/real_iteration_log.jsonl`.

That distinction matters. Judges will reward the idea only if it connects to real KuaiRand-Pure training, GAUC/nDCG@5 evaluation, reproducible benchmark logs, and final submission files.

## What is strong right now

1. **Clear differentiation**
   - The project is not just "AI writes a model."
   - It presents a multi-model research system with separate data processing, cleaning, training, and continual learning stages.

2. **Visible autonomy**
   - The dashboard shows stage-by-stage model responsibilities.
   - It generates requirements, training examples, predictions, and next actions.
   - The benchmark panel shows measured candidate comparisons, promotion logic, and the selected winner.

3. **Good judging alignment**
   - It already speaks the language of the challenge:
     - GAUC
     - nDCG@5
     - benchmark logs
     - manual interventions
     - convergence
     - failure recovery
     - resource awareness

4. **Safe architecture**
   - The OpenAI API key stays on the backend.
   - The browser never directly receives secrets.
   - The OpenAI path has a local fallback so demos do not collapse without a key.

5. **Scalability story**
   - The framework is designed around separating cheap processing/cleaning from expensive model training.
   - The literature strategy argues for retrieval-then-ranking rather than scoring every item directly.

## What is weak right now

1. **The requirement classifier is supporting infrastructure**
   - The browser Naive Bayes model classifies research briefs and does not train the KuaiRand recommender.
   - The README now states clearly that the official KuaiRand pipeline is the submission source of truth.

2. **The dashboard planning loop is not the benchmark**
   - The old fabricated solution-tree metrics have been removed.
   - The panel now reads only the persisted benchmark report and displays measured candidates.

3. **The current challengers do not improve FM**
   - Eleven challenger variants, including PLE multitask variants, were evaluated on validation and all failed the `0.002` promotion margin.
   - The current best model is plain `fm_baseline`, with test primary `0.5953`.

4. **Pairwise ranking needs diagnosis**
   - The raw item feature ablation reaches valid GAUC `0.6387`, matching popularity.
   - Pairwise ranking remains much lower, so the next investigation should focus on objective design, hard-negative selection, and calibration.

### Measured candidate table

| Candidate | Valid primary | Delta vs official `0.6016` | Decision |
|---|---:|---:|---|
| `fm_baseline` | 0.6015 | -0.0001 | selected |
| `fm_plus_ple` | 0.6008 | -0.0008 | rejected |
| `fm_plus_autoscale` | 0.5994 | -0.0022 | rejected |
| `validation_tuned_blend` | 0.5800 | -0.0216 | rejected |
| `time_decay` | 0.5797 | -0.0219 | rejected |
| `multi_behavior` | 0.5786 | -0.0230 | rejected |
| `retrieval_then_rank` | 0.5778 | -0.0238 | rejected |
| `pairwise_uniform` | 0.5663 | -0.0353 | rejected |
| `pairwise_ranking` | 0.5410 | -0.0606 | rejected |
| `auxiliary_multitask` | 0.5339 | -0.0677 | rejected |
| `randomized_exposure_item` | 0.4837 | -0.1179 | rejected |
| `ple_multitask` | 0.4328 | -0.1688 | rejected |

The randomized-exposure candidate uses KuaiRand's `is_rand` rows to build an exposure-aware item statistic. The supplied files do not expose a per-row propensity probability, so this run is a randomized-subset diagnostic rather than a full inverse-propensity or counterfactual estimator.

5. **The current winner is a baseline, not a research improvement**
   - The final artifacts are now generated and schema-validated.
   - The official validation reference is `0.6016` primary; this run's FM validation score is `0.6015` and therefore does not claim a validation improvement.
   - On the held-out test reference, FM is `+0.0007` primary (`0.5953` versus `0.5946`), but that is not evidence of tuning to test because test was evaluated only after validation selection.

## What to build next

### Priority 1: Find out why pairwise_ranking underperforms pop

The raw item feature matches popularity, but the pairwise candidate falls below both. The next work should test pair sampling, warm-start/calibration choices, and per-user score distributions with validation-only ablations:

```text
data.py -> feature builder -> model trainer -> evaluate.py -> submit.py
```

The completed run now produces:

- `runs/benchmark_report.json`
- `outputs/submission_valid.csv`
- `outputs/submission_test.csv`
- `outputs/final_metrics.json`
- `outputs/selected_fm_checkpoint.npz`
- `runs/real_iteration_log.jsonl`

The CSVs were checked with the official starter-kit validator. The iteration log records each candidate's hypothesis, measured validation metrics, delta versus the official baseline, decision, planner order, and intervention count.

### Priority 2: Real feature engineering

Implement the first feature families:

- video popularity
- author popularity
- user long-view rate
- user-author affinity
- duration bucket
- tab-level behavior rate
- time-decayed user history
- hate/short-view negative signal

### Priority 3: Real autonomous experiment loop

Let the learner choose from a safe experiment menu:

- add feature family
- remove feature family
- change target weighting
- try pairwise ranking
- change regularization
- rollback failed experiment

Every iteration should log:

- hypothesis
- code/config diff
- GAUC
- nDCG@5
- primary
- wall-clock
- token estimate
   - promotion decision: accepted/rejected
- recovery event, if any

### Priority 4: Real scalability evidence

Add at least one scalable design element:

- retrieval-then-ranking candidate set
- compact user profiles
- cached feature tables
- streaming CSV processing
- CPU-first model

## Suggested final positioning

Do not pitch the current project as "our final model already improves KuaiRand."

Pitch it as:

> We built AutoScaleRec, a multi-model autonomous recommender research framework. The dashboard shows the requirement pipeline and report-backed candidate comparison. The benchmark pipeline uses KuaiRand-Pure to evaluate each experiment with GAUC and nDCG@5, then keeps only a candidate that improves validation by the promotion margin.

## Go / no-go assessment

| Area | Current status | Competition readiness |
|---|---|---|
| Idea uniqueness | Strong | Ready |
| Dashboard demo | Good | Ready with measured-results caveat |
| Multi-model framework | Good concept + provider routing | Ready with clear scope |
| KuaiRand benchmark | Full-data evaluation completed | Ready for reproducible benchmarking |
| Real model improvement | FM selected; challengers rejected | Improvement not yet proven |
| Submission files | Valid and test CSVs generated and checked with starter kit | Ready |
| README/docs | Updated with measured results and scope | Ready for final polish |

## Bottom line

The project is directionally strong and more differentiated than a normal AI-generated recommender. The benchmark execution and reporting requirements are now covered; the next research work is to improve a challenger on validation while keeping the benchmark-backed dashboard honest.
