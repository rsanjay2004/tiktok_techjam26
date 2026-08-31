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
const requirementList = document.querySelector("#requirement-list");
const trainingList = document.querySelector("#training-list");
const trainingNote = document.querySelector("#training-note");

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
  renderList(requirementList, state.requirements, "No requirements detected yet.");
  renderList(trainingList, state.trainingExamples, "No generated examples yet.");
  trainingNote.textContent = `Scores come from ${trainingCount} training examples after autonomous requirement detection.`;
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

async function fetchRequirementAnalysis(text) {
  try {
    const response = await fetch("/api/analyze-requirements", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ problem: text }),
    });

    if (!response.ok) {
      throw new Error("The local analysis server is not available.");
    }

    return await response.json();
  } catch {
    return localRequirementAnalysis(text);
  }
}

function localRequirementAnalysis(text) {
  const tokens = tokenize(text);
  const categories = {
    "Text Classification": ["classify", "category", "label", "intent", "triage"],
    "Retrieval Assistant": ["retrieve", "search", "documents", "papers", "citations"],
    "Anomaly Detection": ["detect", "unusual", "failure", "risk", "suspicious", "logs"],
    Recommendation: ["recommend", "rank", "suggest", "personalize", "next"],
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
    summary: "The browser fallback detected requirements from keyword evidence and generated retraining examples.",
    requirements: [
      `Classify this as a ${label} problem.`,
      "Extract useful features from the user requirement.",
      "Train the lightweight model with synthetic examples.",
      "Return a measurable prediction and next-step plan.",
    ],
    features,
    trainingExamples: [
      { label, text: `${text} ${features.join(" ")} autonomous requirements evidence` },
      { label, text: `${label} solution needs baseline model evaluation explanation demo` },
    ],
    plan: categoryPlans[label],
  };
}

async function analyze() {
  const text = input.value.trim();
  if (!text) {
    prediction.textContent = "Add a problem";
    confidenceLabel.textContent = "0%";
    confidenceBar.style.width = "0";
    evidenceList.innerHTML = "<li>Paste a research statement to start the autonomous analysis.</li>";
    planList.innerHTML = "<li>The model will generate next actions after classification.</li>";
    scoreboard.innerHTML = "";
    renderAgentState(defaultAgentState, baseTrainingData.length);
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  modelStatus.lastChild.textContent = " Training...";

  const agentState = await fetchRequirementAnalysis(text);
  const generatedExamples = Array.isArray(agentState.trainingExamples) ? agentState.trainingExamples : [];
  const augmentedTrainingData = [...baseTrainingData, ...generatedExamples];
  const trainedModel = trainNaiveBayes(augmentedTrainingData);

  modelStatus.lastChild.textContent = agentState.source.includes("OpenAI") ? " OpenAI trained" : " Local trained";
  render(classify(trainedModel, text), agentState, augmentedTrainingData.length);

  analyzeButton.disabled = false;
  analyzeButton.textContent = "Analyze & train";
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
  analyze();
});
document.querySelector("#sample-button").addEventListener("click", () => {
  const current = samples.shift();
  input.value = current;
  samples.push(current);
  analyze();
});

analyze();
