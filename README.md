# Autonomous ML Research Triage

This project implements **Problem 2: Lightweight Classifier + Dashboard** as an autonomous machine learning research helper.

The app uses an optional OpenAI model as a requirement analyst. It extracts the user's ML requirements, generates synthetic labeled training examples, retrains a small multinomial Naive Bayes classifier, explains the prediction, compares category scores, and creates an experiment plan.

## Why this project works for a demo

- It is autonomous: the user pastes a problem and the system detects requirements, retrains, classifies, explains, and plans without manual category selection.
- It is measurable: each prediction has a confidence score and category comparison.
- It is explainable: the evidence panel shows which terms matched the trained model and the requirement panel shows what the analyst extracted.
- It is robust for demos: if no OpenAI API key is available, a local fallback analyzer still runs.

## How it works

1. The user enters a machine learning problem statement.
2. The backend calls OpenAI Structured Outputs when `OPENAI_API_KEY` is available.
3. The requirement analyst returns a category, requirements, features, generated training examples, and next actions.
4. The browser adds those generated examples to the seed dataset.
5. The lightweight classifier retrains immediately and scores the user's problem.

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

## Suggested pitch

Research teams often lose time deciding which ML method best fits a vague problem. This project acts as an autonomous first-pass research triage model. An OpenAI model detects the requirement, the app automatically retrains a lightweight classifier, and the dashboard turns the raw statement into a classified ML direction, supporting evidence, and a practical experiment plan.

## Future improvements

- Replace the small seed dataset with real labeled hackathon examples.
- Add CSV upload for team-specific training data.
- Track evaluation metrics on a held-out test set.
- Add retrieval over research papers or internal documentation.
- Export the generated plan as a project checklist.
