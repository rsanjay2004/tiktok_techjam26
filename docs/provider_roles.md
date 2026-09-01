# Multi-Provider Agent Research Design

## Evidence-based design

Recent autonomous ML research supports a loop that treats ML engineering as
search over experiments, uses targeted refinement, and learns from execution
feedback. The language model is the planner; the benchmark evaluator is the
source of truth.

- AIDE: https://arxiv.org/abs/2502.13138
- MLE-STAR: https://arxiv.org/abs/2506.15692
- ML-Agent: https://arxiv.org/abs/2505.23723
- MLE-Dojo: https://arxiv.org/abs/2505.07782
- Continual Recommender Systems: https://arxiv.org/abs/2507.03861

## Provider routing

Provider choice should be based on task requirements and measured reliability,
not on a claim that one vendor is always best.

| Stage | Primary responsibility | Model behavior required | Output |
|---|---|---|---|
| Requirement analyst | Convert the brief into a precise ML objective | Strong instruction following and structured output | Task schema and constraints |
| Literature analyst | Find relevant methods and identify transferable ideas | Search/tool use, citation discipline, synthesis | Evidence-backed experiment candidates |
| Data processor | Inspect schema, splits, leakage, and scale | Careful extraction and deterministic reasoning | Dataset profile |
| Data cleaner | Propose quality checks and safe transformations | Conservative reasoning and failure detection | Cleaning plan and warnings |
| Feature researcher | Propose one feature family at a time | Domain reasoning and ablation discipline | Feature experiment |
| Experiment planner | Choose the next safe experiment | Long-horizon planning and use of run memory | Versioned experiment config |
| Code reviewer | Check the proposed change | Precise code analysis and edge-case detection | Review decision |
| Final judge | Explain the result | Independent comparison | Keep, reject, or rollback |

OpenAI, Claude, and Gemini can each fill several roles. The important design is
that every provider receives the same context contract and returns the same JSON
schema. This makes providers replaceable and allows a second provider to act as
an independent reviewer.

## Continuous learning loop

```text
load best checkpoint and experiment memory
  -> propose one change
  -> validate config and data leakage checks
  -> train on KuaiRand-Pure
  -> evaluate GAUC and nDCG@5
  -> measure runtime and memory
  -> compare against the best checkpoint
  -> keep and checkpoint, or reject and rollback
  -> repeat until target, patience, time, or iteration budget is reached
```

The loop must not claim that it is learning simply because an LLM generated a
new suggestion. A real learning event requires a new trained model, a fixed
validation protocol, and an observed reward. The current dashboard implements
the control policy and logging; the real KuaiRand runner supplies the reward
once the full dataset is configured.

## Reward and stopping policy

Use a multi-objective reward rather than score alone:

```text
reward = primary_validation_score
         - latency_penalty
         - memory_penalty
         - training_cost_penalty
```

Stop when any of these is true:

1. The target primary score is reached.
2. Validation has not improved by epsilon for the patience window.
3. The maximum iteration, runtime, token, or compute budget is reached.
4. The same failure repeats and no safe recovery is available.

This gives the system continuous improvement behavior without creating an
unbounded process that can spend resources forever or overfit the validation
split.

## KuaiRand-specific priority

The first experiments should be:

1. Pairwise or listwise ranking aligned with GAUC and nDCG@5.
2. Time-decayed user-history features.
3. Multi-behavior auxiliary signals, with `long_view` retained as the official
   evaluation target.
4. Compact retrieval features followed by a richer ranking stage.
5. Additional model capacity only after the preceding ablations are measured.

Every experiment must be reproducible and log its hypothesis, changed
component, data version, metrics, runtime, resource usage, decision, and
rollback reason.
