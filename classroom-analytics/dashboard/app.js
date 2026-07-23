// Dashboard client: one WebSocket, push-driven UI updates.
const ACTIONS = [
  ["hand_raised", "Hand raised"], ["attentive", "Attentive"],
  ["head_down", "Head down"], ["off_task", "Off-task"],
  ["out_of_seat", "Out of seat"], ["unknown", "Unknown"],
];

const $ = (id) => document.getElementById(id);
const el = {};

function buildCounters() {
  const wrap = $("counters");
  for (const [key, label] of ACTIONS) {
    const c = document.createElement("div");
    c.className = "counter " + key;
    c.innerHTML = `<div class="n" id="c-${key}">0</div><div class="l">${label}</div>`;
    wrap.appendChild(c);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => $("conn-dot").classList.add("live");
  ws.onclose = () => { $("conn-dot").classList.remove("live"); setTimeout(connect, 1500); };
  ws.onmessage = (e) => render(JSON.parse(e.data));
}


function render(msg) {
  const s = msg.stats;
  $("fps").textContent = (msg.fps ?? 0).toFixed(1);
  if (msg.jpeg) {
    const img = $("feed");
    img.src = "data:image/jpeg;base64," + msg.jpeg;
    img.style.display = "block";
    $("feed-placeholder").style.display = "none";
  }
  if (!s) return;

  $("people").textContent = s.num_people;
  const pct = s.attention_pct ?? 0;
  $("attention").innerHTML = `${Math.round(pct)}<i>%</i>`;
  $("ring").style.setProperty("--pct", pct);

  const counts = s.action_counts || {};
  for (const [key] of ACTIONS) $("c-" + key).textContent = counts[key] || 0;

  const alerts = $("alerts");
  alerts.innerHTML = "";
  for (const a of s.alerts || []) {
    const d = document.createElement("div"); d.className = "alert"; d.textContent = a;
    alerts.appendChild(d);
  }

  renderTracks(s.per_track || {}, s.engagement_scores || {});
  drawChart(msg.timeline || []);
}


function renderTracks(perTrack, scores) {
  const wrap = $("tracks");
  wrap.innerHTML = "";
  const ids = Object.keys(perTrack).sort((a, b) => a - b);
  if (!ids.length) { wrap.innerHTML = '<span style="color:var(--mut)">no one tracked</span>'; return; }
  for (const id of ids) {
    const score = Math.round((scores[id] ?? 0) * 100);
    const t = document.createElement("div");
    t.className = "track";
    t.innerHTML =
      `<div class="tid">#${id}</div><div class="act">${perTrack[id].replace(/_/g, " ")}</div>` +
      `<div class="bar"><i style="width:${score}%"></i></div>`;
    wrap.appendChild(t);
  }
}

function drawChart(timeline) {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, pad = 24;
  ctx.clearRect(0, 0, W, H);
  // gridlines at 0/50/100%
  ctx.strokeStyle = "#262c37"; ctx.fillStyle = "#8b93a3"; ctx.font = "11px sans-serif";
  for (const p of [0, 50, 100]) {
    const y = H - pad - (p / 100) * (H - 2 * pad);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - 6, y); ctx.stroke();
    ctx.fillText(p + "%", 2, y + 3);
  }
  if (timeline.length < 2) return;
  const n = timeline.length, x0 = timeline[0][0], x1 = timeline[n - 1][0];
  const span = Math.max(x1 - x0, 1e-6);
  ctx.beginPath(); ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2;
  timeline.forEach(([t, p], i) => {
    const x = pad + ((t - x0) / span) * (W - pad - 6);
    const y = H - pad - (p / 100) * (H - 2 * pad);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

buildCounters();
connect();
