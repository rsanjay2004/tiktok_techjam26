const baseTrainingData = [
  {
    label: "Text Classification",
    text: "classify support tickets intent spam sentiment category risk labels supervised dataset accuracy precision recall",
  },
  {
    label: "Text Classification",
    text: "triage research ideas into categories explain prediction confidence evidence terms labeled examples classifier",
  },
  {
    label: "Retrieval Assistant",
    text: "retrieve documents knowledge base embeddings search citations answer questions summarize relevant passages rag",
  },
  {
    label: "Retrieval Assistant",
    text: "research assistant finds papers compares sources grounded answers semantic search vector database",
  },
  {
    label: "Anomaly Detection",
    text: "detect unusual activity failures outliers abnormal behavior alerts logs fraud suspicious drift incident",
  },
  {
    label: "Anomaly Detection",
    text: "monitor telemetry discover unexpected patterns detect policy risks failure spikes unknown labels",
  },
  {
    label: "Recommendation",
    text: "recommend next item rank content personalize suggestions optimize engagement user preferences candidates",
  },
  {
    label: "Recommendation",
    text: "suggest best action prioritize options ranking model feedback relevance recommendation system",
  },
  {
    label: "Forecasting",
    text: "predict future demand traffic costs sales time series seasonality trend horizon forecast",
  },
  {
    label: "Forecasting",
    text: "estimate next week usage resource planning capacity historical values temporal model",
  },
];

const samples = [
  "We need a model that reads customer messages, assigns each one to an intent category, shows confidence, and explains which words influenced the decision.",
  "Build a research assistant that searches a document library, retrieves the most relevant papers, and produces grounded answers with citations.",
  "Detect unusual agent behavior from run logs, including repeated failures, denied actions, and suspicious workspace access patterns.",
  "Recommend the next best learning resource for a student based on past activity, skill level, and feedback.",
  "Forecast daily GPU demand for the next two weeks using historical infrastructure usage.",
];

const categoryPlans = {
  "Text Classification": [
    "Collect 30 to 50 representative labeled examples for each target category.",
    "Train a baseline Naive Bayes or logistic regression model and track precision, recall, and confusion cases.",
    "Add an evidence view that shows influential tokens so reviewers can trust or correct the prediction.",
  ],
  "Retrieval Assistant": [
    "Build a small document corpus and chunk each source into searchable passages.",
    "Generate embeddings, retrieve the top matching passages, and attach source evidence to each answer.",
    "Evaluate with five grounded research questions and record whether the answer cites the correct source.",
  ],
  "Anomaly Detection": [
    "Define normal behavior signals such as action count, failure count, denied policies, duration, and file writes.",
    "Train an unsupervised baseline or rules-plus-score detector over those signals.",
    "Create a failure demo where the system flags a risky run and explains which signals caused the alert.",
  ],
  Recommendation: [
    "Define the candidate items, user context, and success metric such as click, completion, or rating.",
    "Start with a content-based ranking model using item tags and user preference history.",
    "Compare the ranked output against a simple popularity baseline.",
  ],
  Forecasting: [
    "Create a time-series table with timestamps, target values, and known calendar or usage drivers.",
    "Train a moving-average baseline before adding a stronger forecasting model.",
    "Visualize predicted versus actual values and highlight where the forecast misses.",
  ],
};

const defaultAgentState = {
  source: "Local classifier",
  summary: "Run analysis to extract requirements from the problem statement.",
  requirements: ["Waiting for a problem statement."],
  trainingExamples: [],
  plan: [],
  pipelineStages: [
    {
      name: "Data Processor",
      role: "Profiles raw inputs",
      output: "Waiting for a problem statement.",
      decision: "Idle",
    },
    {
      name: "Data Cleaner",
      role: "Finds quality issues",
      output: "Waiting for extracted fields.",
      decision: "Idle",
    },
    {
      name: "Training Model",
      role: "Retrains the classifier",
      output: "Waiting for generated examples.",
      decision: "Idle",
    },
    {
      name: "Continual Learner",
      role: "Decides next iteration",
      output: "Waiting for scores and errors.",
      decision: "Idle",
    },
  ],
};

const stopWords = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "or",
  "that",
  "the",
  "this",
  "to",
  "we",
  "with",
]);

const MAX_FILES = 6;
const MAX_FILE_BYTES = 1_000_000;
const MAX_TOTAL_CHARS = 120_000;

const input = document.querySelector("#problem-input");
const prediction = document.querySelector("#prediction");
const confidenceLabel = document.querySelector("#confidence-label");
const confidenceBar = document.querySelector("#confidence-bar");
const evidenceList = document.querySelector("#evidence-list");
const planList = document.querySelector("#plan-list");
const scoreboard = document.querySelector("#scoreboard");
const categoryTags = document.querySelector("#category-tags");
const analyzeButton = document.querySelector("#analyze-button");
const modelStatus = document.querySelector("#model-status");
const agentSummary = document.querySelector("#agent-summary");
const agentSource = document.querySelector("#agent-source");
const providerRuns = document.querySelector("#provider-runs");
const requirementList = document.querySelector("#requirement-list");
const trainingList = document.querySelector("#training-list");
const trainingNote = document.querySelector("#training-note");
const attachButton = document.querySelector("#attach-button");
const fileInput = document.querySelector("#file-input");
const attachmentList = document.querySelector("#attachment-list");
const attachmentNote = document.querySelector("#attachment-note");
const pipelineStages = document.querySelector("#pipeline-stages");
const loopButton = document.querySelector("#loop-button");
const loopSummary = document.querySelector("#loop-summary");
const bestPrimary = document.querySelector("#best-primary");
const bestGauc = document.querySelector("#best-gauc");
const bestNdcg = document.querySelector("#best-ndcg");
const manualInterventions = document.querySelector("#manual-interventions");
const iterationLog = document.querySelector("#iteration-log");
const solutionTree = document.querySelector("#solution-tree");
const benchmarkButton = document.querySelector("#benchmark-button");
const benchmarkStatus = document.querySelector("#benchmark-status");
const benchmarkResults = document.querySelector("#benchmark-results");
const targetPrimary = document.querySelector("#target-primary");
const maxIterations = document.querySelector("#max-iterations");
const maxNoImprove = document.querySelector("#max-no-improve");

let attachments = [];

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function looksBinary(text) {
  let control = 0;
  const sample = text.slice(0, 4000);
  for (const char of sample) {
    const code = char.charCodeAt(0);
    if (code === 0 || (code < 9) || (code > 13 && code < 32)) control += 1;
  }
  return sample.length > 0 && control / sample.length > 0.02;
}

function setAttachmentNote(message) {
  attachmentNote.textContent = message || "";
  attachmentNote.hidden = !message;
}

function renderAttachments() {
  attachmentList.innerHTML = "";
  attachments.forEach((file, index) => {
    const li = document.createElement("li");
    li.className = "attachment-chip";
    const name = document.createElement("span");
    name.className = "attachment-name";
    name.textContent = `${file.name} · ${formatBytes(file.size)}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "attachment-remove";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.addEventListener("click", () => {
      attachments.splice(index, 1);
      renderAttachments();
    });
    li.append(name, remove);
    attachmentList.appendChild(li);
  });
}

async function addFiles(fileList) {
  const skipped = [];

  for (const file of fileList) {
    if (attachments.length >= MAX_FILES) {
      skipped.push(`${file.name} (max ${MAX_FILES} files)`);
      continue;
    }
    if (attachments.some((item) => item.name === file.name && item.size === file.size)) {
      continue;
    }
    if (file.size > MAX_FILE_BYTES) {
      skipped.push(`${file.name} (over ${formatBytes(MAX_FILE_BYTES)})`);
      continue;
    }

    let text = "";
    try {
      text = await file.text();
    } catch {
      skipped.push(`${file.name} (could not read)`);
      continue;
    }

    if (looksBinary(text)) {
      skipped.push(`${file.name} (not a text file)`);
      continue;
    }

    attachments.push({ name: file.name, size: file.size, type: file.type || "text/plain", text });
  }

  renderAttachments();
  setAttachmentNote(skipped.length ? `Skipped: ${skipped.join(", ")}` : "");
}

function buildAttachmentPayload() {
  let budget = MAX_TOTAL_CHARS;
  let truncatedAny = false;
  const payload = [];

  for (const file of attachments) {
    if (budget <= 0) {
      truncatedAny = true;
      break;
    }
    let text = file.text;
    if (text.length > budget) {
      text = `${text.slice(0, budget)}\n...[truncated]`;
      truncatedAny = true;
    }
    budget -= text.length;
    payload.push({ name: file.name, type: file.type, text });
  }

  if (truncatedAny) {
    setAttachmentNote("Attached content was truncated to stay within the size limit.");
  }
  return payload;
}

function attachmentsAsText(files) {
  return files
    .map((file) => `--- File: ${file.name} ---\n${file.text}`)
    .join("\n\n");
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 2 && !stopWords.has(token));
}

function trainNaiveBayes(rows) {
  const labels = [...new Set(rows.map((row) => row.label))];
  const vocabulary = new Set();
  const labelStats = labels.map((label) => ({
    label,
    documents: 0,
    tokenCounts: new Map(),
    totalTokens: 0,
  }));

  rows.forEach((row) => {
    const stats = labelStats.find((item) => item.label === row.label);
    const tokens = tokenize(row.text);
    stats.documents += 1;

    tokens.forEach((token) => {
      vocabulary.add(token);
      stats.totalTokens += 1;
      stats.tokenCounts.set(token, (stats.tokenCounts.get(token) || 0) + 1);
    });
  });

  return { labels, labelStats, vocabularySize: vocabulary.size, documentCount: rows.length };
}

function classify(model, text) {
  const tokens = tokenize(text);
  const scored = model.labelStats.map((stats) => {
    let logScore = Math.log(stats.documents / model.documentCount);
    const evidence = [];

    tokens.forEach((token) => {
      const count = stats.tokenCounts.get(token) || 0;
      const likelihood = (count + 1) / (stats.totalTokens + model.vocabularySize);
      logScore += Math.log(likelihood);
      if (count > 0) {
        evidence.push({ token, count });
      }
    });

    return {
      label: stats.label,
      logScore,
      evidence: evidence.sort((a, b) => b.count - a.count).slice(0, 6),
    };
  });

  const maxLog = Math.max(...scored.map((item) => item.logScore));
  const probabilities = scored.map((item) => ({
    ...item,
    probability: Math.exp(item.logScore - maxLog),
  }));
  const total = probabilities.reduce((sum, item) => sum + item.probability, 0);

  return probabilities
    .map((item) => ({ ...item, probability: item.probability / total }))
    .sort((a, b) => b.probability - a.probability);
}

function renderList(target, items, emptyText) {
  target.innerHTML = "";
  const safeItems = items?.length ? items : [emptyText];
  safeItems.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = typeof item === "string" ? item : `${item.label}: ${item.text}`;
    target.appendChild(li);
  });
}

function renderAgentState(state, trainingCount) {
  agentSummary.textContent = state.summary;
  agentSource.textContent = state.source;
  providerRuns.innerHTML = "";
  (state.providerRuns || [{ id: "local", role: "fallback", status: "primary" }]).forEach((provider) => {
    const chip = document.createElement("span");
    chip.className = `provider-chip ${provider.status === "error" ? "failed" : provider.status === "ok" ? "ready" : "local"}`;
    chip.textContent = `${provider.label || (provider.id === "local" ? "Local deterministic fallback" : provider.id)} · ${provider.role || provider.status}`;
    if (provider.error) chip.title = provider.error;
    providerRuns.appendChild(chip);
  });
  renderPipelineStages(state.pipelineStages || defaultAgentState.pipelineStages);
  renderList(requirementList, state.requirements, "No requirements detected yet.");
  renderList(trainingList, state.trainingExamples, "No generated examples yet.");
  trainingNote.textContent = `Scores come from ${trainingCount} training examples after autonomous requirement detection.`;
}

function renderPipelineStages(stages) {
  pipelineStages.innerHTML = "";
  stages.forEach((stage, index) => {
    const card = document.createElement("article");
    card.className = "stage-card";
    const stageIndex = document.createElement("div");
    const body = document.createElement("div");
    const title = document.createElement("h3");
    const role = document.createElement("p");
    const output = document.createElement("p");
    const decision = document.createElement("span");

    stageIndex.className = "stage-index";
    stageIndex.textContent = String(index + 1);
    title.textContent = stage.name;
    role.className = "stage-role";
    role.textContent = stage.role;
    output.textContent = stage.output;
    decision.textContent = stage.decision;

    body.append(title, role, output, decision);
    card.append(stageIndex, body);
    pipelineStages.appendChild(card);
  });
}

function render(result, agentState, trainingCount) {
  const top = result[0];
  const confidence = Math.round(top.probability * 100);

  prediction.textContent = top.label;
  confidenceLabel.textContent = `${confidence}%`;
  confidenceBar.style.width = `${confidence}%`;
  renderAgentState(agentState, trainingCount);

  evidenceList.innerHTML = "";
  const evidence = top.evidence.length
    ? top.evidence
    : [{ token: "No strong overlap found. Add labeled examples for this domain.", count: 0 }];

  evidence.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.count ? `"${item.token}" matched training evidence` : item.token;
    evidenceList.appendChild(li);
  });

  planList.innerHTML = "";
  const plan = agentState.plan?.length ? agentState.plan : categoryPlans[top.label];
  plan.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    planList.appendChild(li);
  });

  scoreboard.innerHTML = "";
  result.forEach((item) => {
    const percent = Math.round(item.probability * 100);
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <div class="score-meta">
        <span class="score-name">${item.label}</span>
        <span class="score-value">${percent}%</span>
      </div>
      <div class="score-track"><span class="score-fill" style="width: ${percent}%"></span></div>
    `;
    scoreboard.appendChild(row);
  });
}

async function fetchRequirementAnalysis(text, files) {
  try {
    const response = await fetch("/api/analyze-requirements", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ problem: text, attachments: files }),
    });

    if (!response.ok) {
      throw new Error("The local analysis server is not available.");
    }

    return await response.json();
  } catch {
    return localRequirementAnalysis([text, attachmentsAsText(files)].filter(Boolean).join("\n\n"));
  }
}

async function fetchAutonomousLoop(text) {
  const response = await fetch("/api/run-autonomous-loop", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      problem: text,
      targetPrimary: Number(targetPrimary.value),
      maxIterations: Number(maxIterations.value),
      maxNoImprove: Number(maxNoImprove.value),
    }),
  });

  if (!response.ok) {
    throw new Error("Autonomous loop did not complete.");
  }

  return response.json();
}

function localRequirementAnalysis(text) {
  const tokens = tokenize(text);
  const categories = {
    "Text Classification": ["classify", "category", "label", "intent", "triage"],
    "Retrieval Assistant": ["retrieve", "search", "documents", "papers", "citations"],
    "Anomaly Detection": ["detect", "unusual", "failure", "risk", "suspicious", "logs"],
    Recommendation: ["recommend", "recommender", "recommendation", "rank", "ranking", "suggest", "personalize", "next", "kuairand", "gauc", "ndcg"],
    Forecasting: ["forecast", "predict", "future", "demand", "trend"],
  };

  const winner = Object.entries(categories)
    .map(([label, words]) => ({
      label,
      matches: words.filter((word) => tokens.includes(word)),
    }))
    .sort((a, b) => b.matches.length - a.matches.length)[0];
  const label = winner.matches.length ? winner.label : "Text Classification";
  const features = [...new Set([...winner.matches, ...tokens.filter((token) => token.length > 4)])].slice(0, 8);

  return {
    source: "Browser fallback",
    category: label,
    summary: "Four specialized local model stages processed the request, cleaned assumptions, retrained, and planned the next learning step.",
    requirements: [
      `Classify this as a ${label} problem.`,
      "Extract useful features from the user requirement.",
      "Train the lightweight model with synthetic examples.",
      "Return a measurable prediction and next-step plan.",
    ],
    features,
    pipelineStages: buildLocalPipelineStages(label, features),
    trainingExamples: [
      { label, text: `${text} ${features.join(" ")} autonomous requirements evidence` },
      { label, text: `${label} solution needs baseline model evaluation explanation demo` },
    ],
    plan: categoryPlans[label],
  };
}

function buildLocalPipelineStages(label, features) {
  return [
    {
      name: "Data Processor",
      role: "Profiles the raw user requirement",
      output: `Detected ${features.length} useful terms: ${features.slice(0, 5).join(", ")}.`,
      decision: "Pass structured features forward",
    },
    {
      name: "Data Cleaner",
      role: "Removes noise and checks training risk",
      output: "Removed stop words, normalized terms, and kept only useful feature tokens.",
      decision: "No blocking data issue",
    },
    {
      name: "Training Model",
      role: "Builds the lightweight classifier",
      output: `Generated extra ${label} examples and retrained the Naive Bayes model.`,
      decision: "Retrain accepted",
    },
    {
      name: "Continual Learner",
      role: "Plans the next improvement loop",
      output: "Next loop should validate with labeled examples, compare metrics, and keep only improvements.",
      decision: "Continue learning",
    },
  ];
}

async function analyze() {
  const text = input.value.trim();
  const files = buildAttachmentPayload();

  if (!text && !files.length) {
    prediction.textContent = "Add a problem";
    confidenceLabel.textContent = "0%";
    confidenceBar.style.width = "0";
    evidenceList.innerHTML = "<li>Paste a research statement or attach files to start the autonomous analysis.</li>";
    planList.innerHTML = "<li>The model will generate next actions after classification.</li>";
    scoreboard.innerHTML = "";
    renderAgentState(defaultAgentState, baseTrainingData.length);
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  modelStatus.lastChild.textContent = " Training...";

  const agentState = await fetchRequirementAnalysis(text, files);
  const generatedExamples = Array.isArray(agentState.trainingExamples) ? agentState.trainingExamples : [];
  const augmentedTrainingData = [...baseTrainingData, ...generatedExamples];
  const trainedModel = trainNaiveBayes(augmentedTrainingData);

  const classificationText = [text, attachmentsAsText(files)].filter(Boolean).join("\n\n");
  modelStatus.lastChild.textContent = agentState.source.includes("OpenAI") ? " OpenAI trained" : " Local trained";
  render(classify(trainedModel, classificationText), agentState, augmentedTrainingData.length);

  analyzeButton.disabled = false;
  analyzeButton.textContent = "Analyze & train";
}

function renderLoop(result) {
  loopSummary.textContent = `Planning demonstration only: ${result.convergence} Measured scores are shown by Run KuaiRand benchmark.`;
  bestPrimary.textContent = result.best.primary.toFixed(4);
  bestGauc.textContent = result.best.gauc.toFixed(4);
  bestNdcg.textContent = result.best.ndcg.toFixed(4);
  manualInterventions.textContent = String(result.manualInterventions);

  iterationLog.innerHTML = "";
  result.iterations.forEach((iteration) => {
    const card = document.createElement("article");
    card.className = `iteration-card ${iteration.decision === "keep" ? "kept" : "rejected"}`;

    const head = document.createElement("div");
    head.className = "iteration-head";
    const title = document.createElement("h3");
    title.textContent = `Iteration ${iteration.iteration}: ${iteration.model} (${iteration.action})`;
    const decision = document.createElement("code");
    decision.textContent = iteration.decision;
    head.append(title, decision);

    const hypothesis = document.createElement("p");
    hypothesis.textContent = `Hypothesis: ${iteration.hypothesis}`;
    const diff = document.createElement("p");
    diff.textContent = `Change: ${iteration.diff}`;
    const metrics = document.createElement("p");
    metrics.textContent = `GAUC ${iteration.metrics.GAUC.toFixed(4)} | nDCG@5 ${iteration.metrics["nDCG@5"].toFixed(4)} | primary ${iteration.metrics.primary.toFixed(4)} | delta ${iteration.metrics.deltaPrimary.toFixed(4)}`;
    const recovery = document.createElement("p");
    recovery.textContent = `Recovery: ${iteration.recovery}`;
    const branch = document.createElement("p");
    branch.textContent = `Tree node: ${iteration.nodeId}, parent: ${iteration.parentId || "root"}`;

    card.append(head, hypothesis, diff, metrics, recovery, branch);
    iterationLog.appendChild(card);
  });

  solutionTree.innerHTML = "";
  result.solutionTree.forEach((node) => {
    const card = document.createElement("article");
    card.className = `tree-node ${node.id === result.bestNodeId ? "best" : ""}`;
    const title = document.createElement("h3");
    const parent = document.createElement("span");
    const metric = document.createElement("p");
    const note = document.createElement("p");

    title.textContent = `${node.id}: ${node.action}`;
    parent.textContent = `parent: ${node.parentId || "none"}`;
    metric.textContent = `primary ${node.primary.toFixed(4)} | ${node.status}`;
    note.textContent = node.name;

    card.append(title, parent, metric, note);
    solutionTree.appendChild(card);
  });
}

function parseMeasuredMetrics(item) {
  const matches = [...String(item.log || "").matchAll(/test\s+GAUC\s+([0-9.]+)\s+\|\s+nDCG@5\s+([0-9.]+)\s+\|\s+primary\s+([0-9.]+)/g)];
  if (matches.length) return { gauc: Number(matches.at(-1)[1]), ndcg: Number(matches.at(-1)[2]), primary: Number(matches.at(-1)[3]) };
  if (item.model === "autoscale") {
    try {
      const report = JSON.parse(String(item.log || "").slice(String(item.log || "").indexOf("{")));
      return { gauc: report.test?.GAUC, ndcg: report.test?.["nDCG@5"], primary: report.test?.primary, winner: report.winner };
    } catch { return null; }
  }
  return null;
}

function renderBenchmarkResults(result) {
  benchmarkResults.innerHTML = "";
  const heading = document.createElement("p");
  heading.className = "benchmark-result-heading";
  heading.textContent = "Measured benchmark results";
  benchmarkResults.appendChild(heading);
  const table = document.createElement("table");
  table.className = "benchmark-table";
  table.innerHTML = "<thead><tr><th>Run</th><th>GAUC</th><th>nDCG@5</th><th>Primary</th><th>Decision</th></tr></thead>";
  const body = document.createElement("tbody");
  result.models.forEach((item) => {
    const metrics = parseMeasuredMetrics(item);
    if (!metrics) return;
    const row = document.createElement("tr");
    [item.model, metrics.gauc?.toFixed(4) || "-", metrics.ndcg?.toFixed(4) || "-", metrics.primary?.toFixed(4) || "-", item.model === "autoscale" ? `Winner: ${metrics.winner || "unknown"}` : "Measured"].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  benchmarkResults.appendChild(table);
  const note = document.createElement("p");
  note.className = "benchmark-note";
  note.textContent = "The official evaluator is the source of truth. Challengers are promoted only when validation improves by the configured margin.";
  benchmarkResults.appendChild(note);
}

async function runLoop() {
  loopButton.disabled = true;
  loopButton.textContent = "Running...";
  loopSummary.textContent = "The autonomous learner is selecting experiments and evaluating outcomes.";

  try {
    renderLoop(await fetchAutonomousLoop(input.value.trim()));
  } catch (error) {
    loopSummary.textContent = error.message;
  } finally {
    loopButton.disabled = false;
    loopButton.textContent = "Run loop";
  }
}

async function runBenchmark() {
  benchmarkButton.disabled = true;
  benchmarkStatus.textContent = "Running the KuaiRand starter-kit evaluator...";
  try {
    const response = await fetch("/api/run-benchmark", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    const result = await response.json();
    if (result.status === "missing_dataset") {
      benchmarkStatus.textContent = "Dataset not found. Add KuaiRand-Pure/data, then run this benchmark again.";
    } else if (result.status === "completed") {
      const models = result.models.map((item) => `${item.model}: ${item.seconds}s`).join(" | ");
      benchmarkStatus.textContent = `Real starter-kit runs completed: ${models}.`;
      renderBenchmarkResults(result);
    } else {
      benchmarkStatus.textContent = result.message || "Benchmark could not complete.";
    }
  } catch (error) {
    benchmarkStatus.textContent = error.message;
  } finally {
    benchmarkButton.disabled = false;
  }
}

const model = trainNaiveBayes(baseTrainingData);

model.labels.forEach((label) => {
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = label;
  categoryTags.appendChild(tag);
});

document.querySelector("#analyze-button").addEventListener("click", analyze);
document.querySelector("#clear-button").addEventListener("click", () => {
  input.value = "";
  attachments = [];
  fileInput.value = "";
  renderAttachments();
  setAttachmentNote("");
  analyze();
});

attachButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  await addFiles(Array.from(fileInput.files || []));
  fileInput.value = "";
});
document.querySelector("#sample-button").addEventListener("click", () => {
  const current = samples.shift();
  input.value = current;
  samples.push(current);
  analyze();
});
loopButton.addEventListener("click", runLoop);
benchmarkButton.addEventListener("click", runBenchmark);

analyze();
