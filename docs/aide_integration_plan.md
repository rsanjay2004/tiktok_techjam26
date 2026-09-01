# AIDE-Inspired Continual Learning Loop

## What we borrow from AIDE

AIDE frames machine learning engineering as **solution-space tree search**:

1. Draft a candidate solution.
2. Run and evaluate it.
3. Debug failed branches.
4. Improve promising branches.
5. Select the best solution as the next base.
6. Repeat until the run budget or convergence rule is hit.

For our project, this is stronger than a simple linear loop because it shows not only the final model, but also the search path that got there.

## How it maps to AutoScaleRec

| AIDE concept | AutoScaleRec implementation |
|---|---|
| Solution generator | The continual learner chooses an experiment from a safe menu |
| Draft node | First candidate improvement over the official FM baseline |
| Improve node | Feature/loss/retrieval changes applied to the best current branch |
| Debug node | Failed or regressing experiment that triggers rollback |
| Evaluator | KuaiRand `evaluate.py` for real GAUC and nDCG@5 |
| Selector | Keep the node only if validation primary improves by more than 0.002 |
| Search tree | Report-backed candidate comparison in `runs/benchmark_report.json` |

## Current implementation

The app now includes an AIDE-style demo loop:

- `POST /api/run-autonomous-loop`
- Produces iteration logs
- Produces a solution tree
- Marks draft, improve, and debug nodes
- Selects the best node
- Rejects a regressing branch and rolls back

The AIDE-style planning idea is now paired with a real KuaiRand training path. The dashboard reports measured candidates from `runs/benchmark_report.json`; it does not invent metric values.

## Real benchmark implementation path

To make the AIDE loop real for Track 2:

1. Add the KuaiRand starter-kit scripts to the repo.
2. Add a config file for each experiment node.
3. Let the continual learner choose one safe experiment at a time.
4. Run training.
5. Run `evaluate.py`.
6. Parse real GAUC and nDCG@5.
7. Keep the child node only if it improves validation primary.
8. Write the winning model scores through `submit.py`.

## Why this is a differentiator

Many teams can ask AI to create a recommender. This design uses AI to run a controlled research process:

- It explores multiple branches.
- It logs failures.
- It rolls back bad changes.
- It records why each experiment was tried.
- It stops under the official convergence rule.
- It separates data processing, cleaning, training, and learning decisions.

That is much closer to autonomous ML engineering than a one-shot AI-generated notebook.
