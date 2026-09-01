# Autonomous ML Research Triage

This project implements **Problem 2: Lightweight Classifier + Dashboard** as an autonomous machine learning research helper.

The app uses an optional multi-provider AI layer as a requirement analyst. OpenAI, Claude, and Gemini can each fill the same structured-analysis contract. The first configured provider is the primary analyst; other configured providers independently review the same request. The browser then generates synthetic labeled training examples and retrains a small multinomial Naive Bayes classifier.

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

See [docs/literature_strategy.md](/Users/livelysan/Documents/ChatGPT/tiktok techjam/docs/literature_strategy.md) for the recommended pivot into **AutoScaleRec**, an autonomous scalable recommender research agent for KuaiRand-Pure.

See [docs/kuairand_starter_kit_test_report.md](/Users/livelysan/Documents/ChatGPT/tiktok techjam/docs/kuairand_starter_kit_test_report.md) for the starter-kit zip smoke test results.

See [docs/multi_model_framework.md](/Users/livelysan/Documents/ChatGPT/tiktok techjam/docs/multi_model_framework.md) for the four-model autonomous learning framework.

See [docs/current_project_review.md](/Users/livelysan/Documents/ChatGPT/tiktok techjam/docs/current_project_review.md) for a critical review of what is strong today and what still needs to become real benchmark work.

See [docs/aide_integration_plan.md](/Users/livelysan/Documents/ChatGPT/tiktok techjam/docs/aide_integration_plan.md) for how the project adapts AIDE-style solution-space tree search.

See [docs/provider_roles.md](/Users/livelysan/Documents/ChatGPT/tiktok techjam/docs/provider_roles.md) for the multi-provider routing and continuous-learning policy.

## Real KuaiRand benchmark bridge

The dashboard now includes a **Run KuaiRand benchmark** action. It invokes `kuairand_runner.py`, which delegates data loading and evaluation to the official starter-kit scripts and writes the result to `runs/benchmark_report.json`.

The starter kit and full dataset must be available locally. From the project root:

```bash
export KUAI_DATA_DIR=/path/to/KuaiRand-Pure/data
export KUAI_KIT_DIR=/path/to/kuairand-starter-kit
export KUAI_PYTHON=/path/to/python-with-numpy
npm start
```

The dashboard will report a missing-dataset state instead of showing simulated loop metrics as benchmark results. The autonomous improvement loop remains a controlled planning demo until each experiment is connected to a real KuaiRand training configuration.
