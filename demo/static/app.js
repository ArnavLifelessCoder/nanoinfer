"use strict";

const $ = (id) => document.getElementById(id);
const state = { model: null, controller: null, info: null, decodeMs: [] };

function setError(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}

function percentile(values, p) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1)))];
}

function format(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function selectModel(key) {
  state.model = key;
  for (const button of $("model").children) {
    button.setAttribute("aria-checked", String(button.dataset.key === key));
  }
  updateCompare();
}

// PyTorch float32 decode at the nearest measured thread count, from results/.
function updateCompare() {
  const baseline = state.info?.pytorch_decode_tokens_per_second || {};
  const counts = Object.keys(baseline).map(Number);
  if (!counts.length) {
    $("compare").textContent = "";
    return;
  }
  const threads = Number($("threads").value);
  const nearest = counts.reduce((a, b) => (Math.abs(b - threads) < Math.abs(a - threads) ? b : a));
  $("compare").textContent = `PyTorch f32 decode at ${nearest} threads: ${format(baseline[nearest])} tok/s`;
}

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    state.info = await response.json();
  } catch {
    $("runtime-badge").textContent = "server not reachable";
    return;
  }
  const { models, max_threads: maxThreads } = state.info;
  $("model").replaceChildren(
    ...models.map((model) => {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "radio";
      button.dataset.key = model.key;
      button.textContent = model.key;
      button.title = model.label;
      button.addEventListener("click", () => selectModel(model.key));
      return button;
    }),
  );
  selectModel(models.some((m) => m.key === "int8") ? "int8" : models[0].key);
  $("threads").max = String(maxThreads);
  $("threads").value = String(Math.min(4, maxThreads));
  $("threads-value").textContent = $("threads").value;
  $("runtime-badge").textContent = `${models.length} model${models.length > 1 ? "s" : ""} ready`;
  updateCompare();
}

function resetOutput(prompt) {
  const text = $("text");
  text.replaceChildren();
  const promptSpan = document.createElement("span");
  promptSpan.className = "prompt-text";
  promptSpan.textContent = prompt;
  const generated = document.createElement("span");
  generated.id = "generated";
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  text.append(promptSpan, generated, cursor);
  state.decodeMs = [];
  for (const id of ["stat-ttft", "stat-prefill", "stat-decode", "stat-latency", "stat-tokens"]) {
    $(id).textContent = "-";
  }
  drawChart();
}

function drawChart() {
  const canvas = $("chart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const styles = getComputedStyle(document.documentElement);
  const values = state.decodeMs;
  const top = Math.max(10, ...values) * 1.15;
  context.strokeStyle = styles.getPropertyValue("--chart-grid");
  context.fillStyle = styles.getPropertyValue("--muted");
  context.font = "11px system-ui";
  for (const fraction of [0.25, 0.5, 0.75, 1]) {
    const y = height - fraction * (height - 14);
    context.beginPath();
    context.moveTo(32, y);
    context.lineTo(width, y);
    context.stroke();
    context.fillText(format(top * fraction, 0), 0, y + 4);
  }
  if (!values.length) return;
  const slot = (width - 36) / Math.max(values.length, 40);
  context.fillStyle = styles.getPropertyValue("--chart");
  values.forEach((value, index) => {
    const barHeight = (value / top) * (height - 14);
    context.fillRect(36 + index * slot, height - barHeight, Math.max(1, slot - 1), barHeight);
  });
}

function onEvent(kind, data, totals) {
  if (kind === "prompt") {
    totals.promptTokens = data.tokens;
  } else if (kind === "prefill") {
    $("stat-ttft").textContent = `${format(data.ms, 0)} ms`;
    $("stat-prefill").textContent = format((1000 * data.tokens) / data.ms);
    $("runtime-badge").textContent = `${state.model} - ${data.threads} threads`;
  } else if (kind === "token") {
    totals.generated += 1;
    // The first token comes from the prefill; later ones from a decode step.
    if (totals.generated > 1) state.decodeMs.push(data.ms);
    $("generated").textContent += data.text;
    const decodeTotal = state.decodeMs.reduce((a, b) => a + b, 0);
    if (state.decodeMs.length) {
      $("stat-decode").textContent = format((1000 * state.decodeMs.length) / decodeTotal);
      $("stat-latency").textContent =
        `${format(percentile(state.decodeMs, 0.5))} / ${format(percentile(state.decodeMs, 0.99))}`;
    }
    $("stat-tokens").textContent = String(totals.generated);
    $("text").scrollTop = $("text").scrollHeight;
    drawChart();
  } else if (kind === "done") {
    $("generated").textContent += data.text;
  } else if (kind === "error") {
    setError(data.message);
  }
}

async function generate() {
  setError("");
  const prompt = $("prompt").value;
  const body = {
    prompt,
    model: state.model,
    threads: Number($("threads").value),
    max_new: Number($("max-new").value),
    temperature: Number($("temperature").value),
    top_k: Number($("top-k").value),
    top_p: Number($("top-p").value),
    seed: Number($("seed").value),
  };
  resetOutput(prompt);
  state.controller = new AbortController();
  $("generate").disabled = true;
  $("stop").disabled = false;
  const totals = { generated: 0, promptTokens: 0 };
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `request failed (${response.status})`);
    }
    // Server-sent events over a POST response: blocks separated by a blank line.
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let split;
      while ((split = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const kind = /^event: (.*)$/m.exec(block)?.[1];
        const data = /^data: (.*)$/m.exec(block)?.[1];
        if (kind && data) onEvent(kind, JSON.parse(data), totals);
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") setError(error.message);
  } finally {
    document.querySelector(".text .cursor")?.remove();
    $("generate").disabled = false;
    $("stop").disabled = true;
    state.controller = null;
  }
}

$("generate").addEventListener("click", generate);
$("stop").addEventListener("click", () => state.controller?.abort());
$("threads").addEventListener("input", () => {
  $("threads-value").textContent = $("threads").value;
  updateCompare();
});
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) generate();
});
window.addEventListener("resize", drawChart);
loadInfo();
drawChart();
