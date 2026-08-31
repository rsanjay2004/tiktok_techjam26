# Multi-Model Autonomous Learning Framework

## Core idea

Instead of using one AI model to do everything, the project uses a **specialized model team**. Each model owns one responsibility and passes structured output to the next model.

This makes the system easier to explain, safer to control, and more scalable than a single all-purpose agent.

## The four-model pipeline

| Stage | Model role | What it receives | What it produces |
|---|---|---|---|
| 1. Data Processor | Understands the raw user requirement and dataset shape | User problem, available fields, metric target | Candidate features, task type, usable signals |
| 2. Data Cleaner | Checks whether the data is usable before training | Candidate features and raw data assumptions | Cleaning actions, rejected fields, leakage warnings |
| 3. Training Model | Trains or retrains the actual ML model | Clean features, labels, generated training examples | Model scores, confidence, validation metrics |
| 4. Continual Learner | Decides how the system improves next | Metrics, failures, logs, resource budget | Keep/reject decision, next experiment, stop/retry action |

## How it works in this app

1. The user enters a machine learning problem.
2. The requirement analyst detects the ML direction.
3. The data processor extracts useful features from the requirement.
4. The cleaner removes weak/noisy terms and checks training risk.
5. The training stage generates extra labeled examples and retrains the classifier.
6. The continual learner proposes the next experiment.
7. The dashboard shows the final prediction, evidence, training examples, and stage-by-stage reasoning.

## How this maps to KuaiRand

For the real Track 2 benchmark, the same architecture becomes:

| Stage | KuaiRand implementation |
|---|---|
| Data Processor | Loads KuaiRand-Pure logs, extracts user/video/time/behavior features |
| Data Cleaner | Handles missing values, repeated user-video pairs, invalid labels, and feature leakage |
| Training Model | Trains the recommender and outputs scores for each logged impression |
| Continual Learner | Reads GAUC/nDCG@5, chooses the next experiment, logs the result, and stops on convergence |

## Why this stands out

Most teams can use AI to generate a model once. This framework uses AI to run the **research process**:

- It separates responsibilities.
- It records what each stage decided.
- It can recover from a failed stage.
- It can scale because expensive training is isolated from cheap processing and cleaning.
- It supports continuous improvement through logged experiments.

## Practical safety rule

The AI planner should not be allowed to make unlimited arbitrary changes. It should choose from a controlled experiment menu:

- Add or remove a feature family
- Change the training target or sample weight
- Try a pairwise/listwise ranking loss
- Add time-decayed user history
- Add multi-behavior satisfaction features
- Keep or reject the change based on validation score

That gives the project autonomy without chaos.
