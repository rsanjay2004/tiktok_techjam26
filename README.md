# Autonomous ML Research Triage

This project implements a requirement-triage interface and a KuaiRand-Pure autonomous recommender research benchmark.

## Scope

This submission targets the KuaiRand-Pure recommendation-system track. The requirement classifier is a supporting interface that translates a research brief into candidate experiments; the official KuaiRand pipeline remains the source of truth for training, evaluation, and model selection.

The app uses an optional multi-provider AI layer to interpret research requirements, while the Python pipeline performs the real KuaiRand recommendation experiments. The official evaluator, not an LLM or the dashboard planning demo, determines benchmark quality.

## Why this project works for a demo

- It is autonomous: the user pastes a problem and the system detects requirements, retrains, classifies, explains, and plans without manual category selection.
- It is measurable: each prediction has a confidence score and category comparison.
- It is explainable: the evidence panel shows which terms matched the trained model and the requirement panel shows what the analyst extracted.
- It is robust for demos: if no OpenAI API key is available, a local fallback analyzer still runs.

## How it works

1. The user enters a machine learning problem statement.
2. The backend calls every configured provider using the same analysis contract.
3. The first successful provider is primary; other successful providers are reviewers and are shown in the dashboard.
4. The requirement analyst returns a category, requirements, features, generated training examples, and next actions.
5. The browser adds those generated examples to the seed dataset.
6. The lightweight classifier retrains immediately and scores the user's problem.

## Research categories

- Text Classification
- Retrieval Assistant
- Anomaly Detection
- Recommendation
- Forecasting

## Demo script

1. Run `npm start`.
2. Open `http://localhost:3000`.
3. Paste a new machine learning problem statement or click **Load sample**.
4. Click **Analyze & train**.
5. Explain the detected requirements, generated training examples, predicted category, confidence score, evidence words, and autonomous next actions.
6. Show a fallback case by running without `OPENAI_API_KEY`, then explain that the system still retrains using local requirement detection.

## Optional OpenAI setup

```bash
OPENAI_API_KEY=your-api-key npm start
```

You can also choose a model:

```bash
OPENAI_API_KEY=your-api-key OPENAI_MODEL=gpt-4.1-mini npm start
```

Optional independent providers:

```bash
ANTHROPIC_API_KEY=your-api-key ANTHROPIC_MODEL=claude-3-5-haiku-latest npm start
GEMINI_API_KEY=your-api-key GEMINI_MODEL=gemini-2.0-flash npm start
```

Use all providers together by exporting all three variables before `npm start`. API keys stay on the server and are never sent to the browser. Provider responses only plan and review; the Python KuaiRand evaluator decides whether a trained recommender actually improved.

## Suggested pitch

Research teams often lose time deciding which ML method best fits a vague problem. This project acts as an autonomous first-pass research triage model. An OpenAI model detects the requirement, the app automatically retrains a lightweight classifier, and the dashboard turns the raw statement into a classified ML direction, supporting evidence, and a practical experiment plan.

## Future improvements

- Replace the small seed dataset with real labeled hackathon examples.
- Add CSV upload for team-specific training data.
- Track evaluation metrics on a held-out test set.
- Add retrieval over research papers or internal documentation.
- Export the generated plan as a project checklist.

## Literature-backed competition strategy

See [docs/literature_strategy.md](docs/literature_strategy.md) for the recommended pivot into **AutoScaleRec**, an autonomous scalable recommender research agent for KuaiRand-Pure.

See [docs/kuairand_starter_kit_test_report.md](docs/kuairand_starter_kit_test_report.md) for the starter-kit zip smoke test results.

See [docs/multi_model_framework.md](docs/multi_model_framework.md) for the four-model autonomous learning framework.

See [docs/current_project_review.md](docs/current_project_review.md) for a critical review of the measured results and remaining work.

See [docs/aide_integration_plan.md](docs/aide_integration_plan.md) for how the project adapts AIDE-style solution-space tree search.

See [docs/provider_roles.md](docs/provider_roles.md) for the multi-provider routing and continuous-learning policy.

## Real KuaiRand benchmark bridge

The dashboard now includes a **Run KuaiRand benchmark** action. It invokes `kuairand_runner.py`, which delegates data loading and evaluation to the official starter-kit scripts and writes the result to `runs/benchmark_report.json`.

The starter kit and full dataset must be available locally. From the project root:

```bash
export KUAI_DATA_DIR=/path/to/KuaiRand-Pure/data
export KUAI_KIT_DIR=/path/to/kuairand-starter-kit
export KUAI_PYTHON=/path/to/python-with-numpy
npm start
```

The dashboard separates the planning demonstration from measured benchmark output and displays the real model scores and selected winner directly.

## Reproducible benchmark outputs

The full-data benchmark writes these files after a successful run:

- `runs/benchmark_report.json`: starter-kit run status, timing, and model logs.
- `runs/real_iteration_log.jsonl`: one measured record per candidate, including hypothesis, validation metrics, baseline delta, decision, and intervention count.
- `outputs/submission_valid.csv`: schema-validated validation submission.
- `outputs/submission_test.csv`: schema-validated test submission for final evaluation.
- `outputs/selected_fm_checkpoint.npz`: selected FM parameters (`V`, `W`, and `b`).
- `outputs/final_metrics.json`: winner, validation/test metrics, official-baseline deltas, iteration count, wall-clock time, token count, and GPU hours.
- `runs/experiment_memory.json`: the current best candidate, rejected candidates, promotion margin, and the next experiment selected from prior evidence.

The current full-data run evaluates 12 candidates. It includes hard-negative and uniform-negative pairwise training, a randomized-exposure feature diagnostic, multi-behavior and PLE candidates, and FM promotion with a validation margin of `0.002`. With no `OPENAI_API_KEY`, candidate ordering uses the deterministic safe fallback; with a key, the optional planner may reorder the controlled candidate menu. The benchmark evaluator remains authoritative.

The continual-learning controller is bounded: it reads experiment memory, proposes one controlled change, trains it, evaluates validation metrics, saves a checkpoint only after promotion, and records rollback when the candidate fails. Pairwise training uses a uniform/hard-negative mixture because the measured pure hard-negative variant was unstable. Re-run the full benchmark after code changes to refresh the generated metrics and memory files.
