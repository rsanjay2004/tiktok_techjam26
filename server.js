const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");

const root = __dirname;
const port = Number(process.env.PORT || 3000);
const openAiKey = process.env.OPENAI_API_KEY;
const openAiModel = process.env.OPENAI_MODEL || "gpt-4.1-mini";

const categories = [
  "Text Classification",
  "Retrieval Assistant",
  "Anomaly Detection",
  "Recommendation",
  "Forecasting",
];

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 60_000) {
        request.destroy();
        reject(new Error("Request body is too large."));
      }
    });
    request.on("end", () => resolve(body));
    request.on("error", reject);
  });
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2);
}

function fallbackRequirementAnalysis(problem) {
  const tokens = new Set(tokenize(problem));
  const keywordMap = [
    ["Anomaly Detection", ["anomaly", "unusual", "failure", "risk", "suspicious", "alert", "detect", "logs"]],
    ["Retrieval Assistant", ["retrieve", "search", "documents", "papers", "citations", "knowledge", "rag"]],
    ["Text Classification", ["classify", "category", "label", "intent", "sentiment", "triage"]],
    ["Recommendation", ["recommend", "rank", "personalize", "suggest", "next", "preference"]],
    ["Forecasting", ["forecast", "predict", "future", "demand", "time", "series", "trend"]],
  ];

  const scored = keywordMap
    .map(([category, words]) => ({
      category,
      score: words.filter((word) => tokens.has(word)).length,
      words: words.filter((word) => tokens.has(word)),
    }))
    .sort((a, b) => b.score - a.score);

  const top = scored[0].score > 0 ? scored[0] : { category: "Text Classification", words: ["problem", "requirements"] };
  const extracted = [...tokens].filter((token) => token.length > 4).slice(0, 8);
  const requirements = [
    `Primary ML direction: ${top.category}`,
    "Needs an explainable baseline model",
    "Needs measurable confidence or quality evidence",
    "Should produce autonomous next-step recommendations",
  ];

  return {
    source: "Local fallback",
    category: top.category,
    summary: "The local analyzer inferred requirements from keyword evidence and generated extra training rows.",
    requirements,
    features: [...new Set([...top.words, ...extracted])].slice(0, 10),
    trainingExamples: [
      {
        label: top.category,
        text: `${problem} ${top.words.join(" ")} explainable confidence requirements autonomous baseline`,
      },
      {
        label: top.category,
        text: `${top.category} project with ${extracted.join(" ")} evaluation evidence demo plan`,
      },
    ],
    plan: [
      "Validate the detected category with 10 hand-labeled problem statements.",
      "Retrain the lightweight classifier with the generated examples.",
      "Compare the new prediction with the pre-training baseline.",
    ],
  };
}

function extractOpenAiJson(payload) {
  if (typeof payload.output_text === "string") {
    return JSON.parse(payload.output_text);
  }

  const message = payload.output?.find((item) => item.type === "message");
  const content = message?.content?.find((item) => item.type === "output_text" || item.type === "text");
  if (content?.text) {
    return JSON.parse(content.text);
  }

  throw new Error("OpenAI response did not contain structured JSON text.");
}

async function analyzeWithOpenAi(problem) {
  if (!openAiKey) {
    return fallbackRequirementAnalysis(problem);
  }

  const schema = {
    type: "object",
    additionalProperties: false,
    properties: {
      category: { type: "string", enum: categories },
      summary: { type: "string" },
      requirements: { type: "array", minItems: 3, maxItems: 6, items: { type: "string" } },
      features: { type: "array", minItems: 4, maxItems: 10, items: { type: "string" } },
      trainingExamples: {
        type: "array",
        minItems: 2,
        maxItems: 4,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            label: { type: "string", enum: categories },
            text: { type: "string" },
          },
          required: ["label", "text"],
        },
      },
      plan: { type: "array", minItems: 3, maxItems: 5, items: { type: "string" } },
    },
    required: ["category", "summary", "requirements", "features", "trainingExamples", "plan"],
  };

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      authorization: `Bearer ${openAiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: openAiModel,
      input: [
        {
          role: "system",
          content:
            "You are an autonomous ML research analyst. Extract requirements from the user problem, choose the best ML direction, and generate concise synthetic training examples for a lightweight classifier.",
        },
        { role: "user", content: problem },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "ml_requirement_analysis",
          strict: true,
          schema,
        },
      },
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI request failed: ${response.status} ${errorBody}`);
  }

  return { source: `OpenAI ${openAiModel}`, ...extractOpenAiJson(await response.json()) };
}

async function handleAnalyze(request, response) {
  try {
    const body = JSON.parse(await readBody(request));
    const problem = String(body.problem || "").trim();

    if (!problem) {
      sendJson(response, 400, { error: "A problem statement is required." });
      return;
    }

    sendJson(response, 200, await analyzeWithOpenAi(problem));
  } catch (error) {
    sendJson(response, 200, {
      ...fallbackRequirementAnalysis(""),
      source: "Local fallback",
      summary: `OpenAI analysis was unavailable, so the project used the local fallback. ${error.message}`,
    });
  }
}

async function serveStatic(request, response) {
  const url = new URL(request.url, `http://${request.headers.host}`);
  const requestedPath = url.pathname === "/" ? "/index.html" : url.pathname;
  const filePath = path.normalize(path.join(root, requestedPath));

  if (!filePath.startsWith(root)) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }

  try {
    const file = await fs.readFile(filePath);
    const contentType = mimeTypes[path.extname(filePath)] || "application/octet-stream";
    response.writeHead(200, { "content-type": contentType });
    response.end(file);
  } catch {
    response.writeHead(404);
    response.end("Not found");
  }
}

const server = http.createServer((request, response) => {
  if (request.method === "POST" && request.url === "/api/analyze-requirements") {
    handleAnalyze(request, response);
    return;
  }

  serveStatic(request, response);
});

server.listen(port, () => {
  console.log(`Autonomous ML Research Triage running at http://localhost:${port}`);
  console.log(openAiKey ? `OpenAI requirement detection enabled with ${openAiModel}` : "Using local fallback analysis.");
});
