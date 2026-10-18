// Client holds all "session" state (tick, injected failures, toggle) and
// sends it with every request -- the backend is a pure function of what it's
// given, which is what makes it safe to run as a stateless serverless
// function instead of a persistent Flask/Mininet process.
let network = null;
let tick = 0;
let startTime = Date.now();
let injectedFailures = new Set();
let predictiveFailoverEnabled = true;

document.addEventListener('DOMContentLoaded', () => {
  initNetwork();
  setInterval(updateUptime, 1000);
  pollState();
  setInterval(pollState, 2500);

  document.getElementById('failoverToggle').addEventListener('change', (e) => {
    predictiveFailoverEnabled = e.target.checked;
  });
  document.getElementById('injectBtn').addEventListener('click', () => {
    const link = document.getElementById('linkSelect').value;
    if (link) injectedFailures.add(link);
  });
  document.getElementById('restoreOneBtn').addEventListener('click', () => {
    const link = document.getElementById('linkSelect').value;
    if (link) injectedFailures.delete(link);
  });
  document.getElementById('restoreBtn').addEventListener('click', () => {
    injectedFailures.clear();
  });
  document.getElementById('resetBtn').addEventListener('click', () => {
    tick = 0;
    startTime = Date.now();
    injectedFailures.clear();
  });
});

function initNetwork() {
  const container = document.getElementById('network');
  const options = {
    physics: { enabled: false },
    nodes: {
      shape: 'box', margin: 10, widthConstraint: { maximum: 70 },
      font: { color: '#e0e6ed', face: 'Monaco, monospace', size: 12 },
      color: { background: '#0f1629', border: '#00ffaa', highlight: { background: '#1a2040', border: '#00dd88' } },
    },
    edges: {
      width: 3,
      font: { color: '#a0aec0', size: 10, face: 'Monaco, monospace' },
      smooth: { type: 'continuous', forceDirection: 'none' },
    },
    interaction: { hover: true, navigationButtons: true },
  };
  network = new vis.Network(container, { nodes: [], edges: [] }, options);
}

function updateUptime() {
  const secs = Math.floor((Date.now() - startTime) / 1000);
  const h = String(Math.floor(secs / 3600)).padStart(2, '0');
  const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  document.getElementById('uptime').textContent = `${h}:${m}:${s}`;
}

async function pollState() {
  tick += 1;
  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tick,
        injectedFailures: Array.from(injectedFailures),
        predictiveFailoverEnabled,
      }),
    });
    const state = await res.json();
    render(state);
  } catch (err) {
    console.error('Poll failed:', err);
  }
}

function render(state) {
  renderNetwork(state.nodes, state.edges);
  renderLinkSelect(state.edges);
  renderStats(state.stats);
  renderComparison(state.comparison);
  renderPredictions(state.predictions);
}

function renderNetwork(nodes, edges) {
  const visNodes = nodes.map(n => ({
    id: n.id, label: n.label, x: n.x, y: n.y, fixed: true,
    shape: n.type === 'cloud' ? 'ellipse' : 'box',
  }));
  const visEdges = edges.map(e => ({
    id: e.id, from: e.source, to: e.target, label: e.label,
    color: { color: e.color }, width: e.width,
  }));
  network.setData({ nodes: visNodes, edges: visEdges });
}

function renderLinkSelect(edges) {
  const select = document.getElementById('linkSelect');
  const current = select.value;
  select.innerHTML = edges.map(e => {
    const failed = injectedFailures.has(e.label) ? ' (failed)' : '';
    return `<option value="${e.label}">${e.label}${failed}</option>`;
  }).join('');
  if (current) select.value = current;
}

function renderStats(stats) {
  const cards = [
    ['Active Predictions', stats.active_predictions],
    ['Degraded / Total Links', `${stats.degraded_links} / ${stats.total_links}`],
    ['Avg Latency', `${stats.avg_latency_ms} ms`],
    ['Avg Utilization', `${stats.avg_utilization_pct}%`],
    ['Avg Packet Loss', `${stats.avg_packet_loss_pct}%`],
    ['Throughput', `${stats.current_throughput_mbps} Mbps`],
  ];
  document.getElementById('stats').innerHTML = cards.map(([label, value]) => `
    <div class="stat-card"><div class="label">${label}</div><div class="value">${value}</div></div>
  `).join('');
}

function renderComparison(cmp) {
  document.getElementById('comparison').innerHTML = `
    <div class="comparison-col">
      <h3>WITH PREDICTION</h3>
      <div class="row"><span>Downtime</span><span>${cmp.with_prediction.downtime_ms} ms</span></div>
      <div class="row"><span>Packet Loss</span><span>${cmp.with_prediction.packet_loss_pct}%</span></div>
    </div>
    <div class="comparison-col">
      <h3>WITHOUT PREDICTION</h3>
      <div class="row"><span>Downtime</span><span>${cmp.without_prediction.downtime_ms} ms</span></div>
      <div class="row"><span>Packet Loss</span><span>${cmp.without_prediction.packet_loss_pct}%</span></div>
    </div>
  `;
  document.querySelector('.stats-panel').querySelector('.improvement')?.remove();
  const note = document.createElement('div');
  note.className = 'improvement';
  note.textContent = `${cmp.improvement_pct}% less downtime with predictive failover`;
  document.getElementById('comparison').after(note);
}

function renderPredictions(predictions) {
  const el = document.getElementById('predictions');
  if (!predictions.length) {
    el.innerHTML = '<div class="empty">No active risk predictions.</div>';
    return;
  }
  el.innerHTML = predictions.slice().reverse().map(p => `
    <div class="entry">
      <span class="link">${p.link}</span> — ${p.risk} (confidence ${(p.confidence * 100).toFixed(1)}%)
      <div class="meta">util ${p.metrics.utilization_pct}% · loss ${(p.metrics.packet_loss_frac * 100).toFixed(1)}% · latency ${p.metrics.latency_ms}ms</div>
    </div>
  `).join('');
}
