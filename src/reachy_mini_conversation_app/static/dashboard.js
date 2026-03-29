/* Dashboard JS — status polling, SSE logs, controls */

// ---- Tabs ----
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---- Status polling ----
const LEVEL_ORDER = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 };

function formatUptime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0 ? `${h}h ${m}m ${sec}s` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function setVal(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = "stat-value" + (cls ? " " + cls : "");
}

async function pollHealth() {
  try {
    const resp = await fetch("/api/health?_=" + Date.now());
    if (!resp.ok) return;
    const d = await resp.json();

    document.getElementById("uptime").textContent = "Uptime: " + formatUptime(d.uptime_s);

    setVal("s-provider", d.provider || "—");

    const model = d.model || "—";
    const modelEl = document.getElementById("s-model");
    modelEl.textContent = model;
    modelEl.className = "stat-value" + (model.length > 20 ? " stat-value-sm" : "");

    setVal("s-multimodal", d.multimodal ? "Yes" : "No", d.multimodal ? "ind-green" : "ind-gray");
    setVal("s-camera", d.camera?.active ? "Active" : "No frame", d.camera?.active ? "ind-green" : "ind-yellow");

    const mv = d.movement || {};
    setVal("s-listening", mv.is_listening ? "Yes" : "No", mv.is_listening ? "ind-green" : "ind-gray");
    setVal("s-breathing", mv.breathing_active ? "Yes" : "No", mv.breathing_active ? "ind-green" : "ind-gray");
    setVal("s-queue", String(mv.queue_size ?? "—"));

    const hz = mv.loop_frequency?.last;
    if (hz != null) {
      const color = hz > 80 ? "ind-green" : hz > 50 ? "ind-yellow" : "ind-red";
      setVal("s-hz", hz.toFixed(0), color);
    }
  } catch (e) {
    /* offline */
  }
}

setInterval(pollHealth, 2000);
pollHealth();

// ---- Controls ----
const ctrlStatus = document.getElementById("ctrl-status");

function flashStatus(msg, ok) {
  ctrlStatus.textContent = msg;
  ctrlStatus.className = "status " + (ok ? "ok" : "error");
  setTimeout(() => { ctrlStatus.textContent = ""; }, 3000);
}

document.querySelectorAll("[data-move]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const action = btn.dataset.move;
    const name = btn.dataset.name;
    try {
      const resp = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, name }),
      });
      const d = await resp.json();
      flashStatus(d.status || d.error, resp.ok);
    } catch (e) {
      flashStatus("Request failed", false);
    }
  });
});

document.querySelectorAll("[data-listen]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const enable = btn.dataset.listen === "true";
    try {
      const resp = await fetch("/api/listen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enable }),
      });
      const d = await resp.json();
      flashStatus(d.status || d.error, resp.ok);
    } catch (e) {
      flashStatus("Request failed", false);
    }
  });
});

// ---- Logs SSE ----
const logContainer = document.getElementById("log-container");
const logFilter = document.getElementById("log-level-filter");
const logAutoScroll = document.getElementById("log-autoscroll");
const logClear = document.getElementById("log-clear");

let evtSource = null;
let logLines = [];

function connectLogs() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource("/api/logs");
  evtSource.onmessage = (event) => {
    try {
      const entry = JSON.parse(event.data);
      addLogLine(entry);
    } catch (e) {}
  };
  evtSource.onerror = () => {
    // Reconnect after a delay
    evtSource.close();
    setTimeout(connectLogs, 3000);
  };
}

function addLogLine(entry) {
  const minLevel = logFilter.value;
  if (minLevel !== "ALL" && (LEVEL_ORDER[entry.level] || 0) < (LEVEL_ORDER[minLevel] || 0)) {
    return;
  }

  const div = document.createElement("div");
  div.className = "log-line log-" + entry.level;
  const ts = new Date(entry.ts * 1000);
  const timeStr = ts.toLocaleTimeString("en-GB", { hour12: false });
  div.textContent = `${timeStr} [${entry.level}] ${entry.msg}`;
  logContainer.appendChild(div);
  logLines.push(div);

  // Cap at 1000 visible lines
  while (logLines.length > 1000) {
    const old = logLines.shift();
    old.remove();
  }

  if (logAutoScroll.checked) {
    logContainer.scrollTop = logContainer.scrollHeight;
  }
}

logClear.addEventListener("click", () => {
  logContainer.innerHTML = "";
  logLines = [];
});

connectLogs();

// ---- Config ----
async function loadConfig() {
  try {
    const resp = await fetch("/api/config?_=" + Date.now());
    if (!resp.ok) return;
    const cfg = await resp.json();
    const table = document.getElementById("config-table");
    table.innerHTML = "";

    const labels = {
      provider: "Provider",
      model_name: "Model",
      llm_base_url: "LLM URL",
      tts_base_url: "TTS URL",
      tts_model: "TTS Model",
      asr_base_url: "ASR URL",
      asr_model: "ASR Model",
      llm_multimodal: "Multimodal",
      head_tracker: "Head Tracker",
      custom_profile: "Profile",
      ha_url: "Home Assistant",
    };

    for (const [key, label] of Object.entries(labels)) {
      const val = cfg[key] || "—";
      const k = document.createElement("div");
      k.className = "config-key";
      k.textContent = label;
      const v = document.createElement("div");
      v.className = "config-val";
      v.textContent = val;
      table.appendChild(k);
      table.appendChild(v);
    }
  } catch (e) {}
}

// Load config when tab is clicked
document.querySelector('[data-tab="config"]').addEventListener("click", loadConfig);
