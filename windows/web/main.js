/**
 * Merlin · The Tower — main UI controller.
 * Connects WebSocket, drives the orb + transcript + status, manages
 * the settings drawer, command palette, and spellbook.
 */

import { Orb } from '/static/orb.js';

// ============================================================
// State
// ============================================================
const state = {
  status: 'idle',
  health: null,
  settings: null,
  voices: null,
  startedAt: Date.now(),
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const orb = new Orb($('#orb'));
const transcriptEl = $('#transcript');
const statePill = $('.state-pill');
const tagline = $('[data-testid="status-text"]');

// ============================================================
// State machine — drives orb + body[data-state]
// ============================================================
function setState(s) {
  state.status = s;
  document.body.setAttribute('data-state', s);
  statePill.textContent = s;
  orb.setState(s);
  const taglines = {
    idle: 'awake',
    listening: 'listening',
    thinking: 'pondering',
    speaking: 'speaking',
    muted: 'muted',
  };
  tagline.textContent = taglines[s] || 'awake';
}

// ============================================================
// WebSocket with reconnect
// ============================================================
let ws = null;
let wsBackoff = 500;
let ghostBubble = null;

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    wsBackoff = 500;
    addSystemBubble('connected');
  };
  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      handleEvent(msg);
    } catch (err) { /* ignore malformed */ }
  };
  ws.onclose = () => {
    setState('muted');
    setTimeout(connectWs, wsBackoff);
    wsBackoff = Math.min(wsBackoff * 2, 8000);
  };
  ws.onerror = () => { try { ws.close(); } catch {} };
}

// ============================================================
// Bus event handlers
// ============================================================
function handleEvent(msg) {
  const { type, data } = msg;
  switch (type) {
    case 'hello':
      state.health = data;
      renderStatus();
      renderBootBanner(data.boot, data.subsystems_ready);
      break;
    case 'audio_rms':
      orb.pushRms(data.rms);
      drawVu(data.rms, data.threshold, data.onset);
      break;
    case 'vad_start':
      if (state.status !== 'thinking' && state.status !== 'speaking') {
        setState('listening');
      }
      addGhostBubble('…');
      break;
    case 'vad_end':
      // Wait for stt_complete to solidify
      break;
    case 'stt_start':
      setState('listening');
      break;
    case 'stt_complete':
      removeGhost();
      if (data.text) addBubble('you', data.text, { latency_ms: data.latency_ms });
      break;
    case 'thinking_start':
      setState('thinking');
      addGhostBubble('thinking…', 'merlin');
      break;
    case 'thinking_complete':
      removeGhost();
      addBubble('merlin', data.text, { latency_ms: data.latency_ms, intent: data.intent });
      setState('speaking');
      break;
    case 'thinking_failed':
      removeGhost();
      addSystemBubble('thinking failed');
      setState('idle');
      break;
    case 'tts_start':
      setState('speaking');
      break;
    case 'tts_envelope':
      orb.pushEnvelope(data.envelope, data.duration);
      break;
    case 'tts_complete':
      setState('idle');
      break;
    case 'merlin_speaks':
      // already handled by thinking_complete; greeting path bypasses brain
      if (data.source === 'greeting') addBubble('merlin', data.text, { greeting: true });
      break;
    case 'mute_toggled':
      setState(data.muted ? 'muted' : 'idle');
      break;
    case 'face_arrived':
      addSystemBubble('face arrived');
      break;
    case 'face_lost':
      addSystemBubble('face left');
      break;
    case 'history_cleared':
      transcriptEl.innerHTML = '<div class="empty">history cleared.</div>';
      break;
    case 'settings_updated':
      // No need to refresh full settings — keep things zippy
      break;
    case 'boot_progress': {
      const { stage, status, detail } = data;
      updateBootPill(stage, status, detail);
      const icon = status === 'ready' ? '✓' : status === 'failed' ? '✗' : '…';
      addSystemBubble(`${icon} ${stage}: ${status}${detail ? ' — ' + detail : ''}`);
      break;
    }
    case 'subsystems_ready':
      addSystemBubble('— ready —');
      hideBootBanner();
      pollHealth();
      break;
    case 'user_message': {
      // Server-confirmed user input. If we already added an optimistic
      // bubble for this exact text, mark it confirmed; otherwise add it.
      confirmOrAddUserBubble(data.text);
      break;
    }
    case 'system_message': {
      addSystemBubble(`${data.level === 'error' ? '✗' : data.level === 'warn' ? '!' : 'ℹ'} ${data.text}`);
      break;
    }
  }
}

// ============================================================
// Transcript
// ============================================================
function ensureNotEmpty() {
  const empty = transcriptEl.querySelector('.empty');
  if (empty) empty.remove();
}

function addBubble(role, text, meta = {}) {
  ensureNotEmpty();
  const b = document.createElement('div');
  b.className = `bubble ${role}`;
  b.dataset.testid = `bubble-${role}`;
  if (meta.pending) b.classList.add('pending');
  b.dataset.text = text;
  b.textContent = text;
  if (meta.latency_ms != null) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.innerHTML = `<span>${formatTime(Date.now())}</span><span>${meta.latency_ms} ms</span>`;
    b.appendChild(m);
  }
  transcriptEl.appendChild(b);
  scrollTranscript();
  return b;
}

// When the user types or clicks a chip, show the bubble immediately.
// The server later emits user_message and we mark it confirmed.
function addOptimisticUserBubble(text) {
  return addBubble('you', text, { pending: true });
}

function confirmOrAddUserBubble(text) {
  const pending = transcriptEl.querySelectorAll('.bubble.you.pending');
  for (const b of pending) {
    if (b.dataset.text === text) {
      b.classList.remove('pending');
      return;
    }
  }
  // No matching optimistic bubble — server-originated user message.
  addBubble('you', text);
}

function addGhostBubble(text, role = 'you') {
  ensureNotEmpty();
  removeGhost();
  ghostBubble = document.createElement('div');
  ghostBubble.className = `bubble ${role} ghost`;
  ghostBubble.textContent = text;
  transcriptEl.appendChild(ghostBubble);
  scrollTranscript();
}

function removeGhost() {
  if (ghostBubble) {
    ghostBubble.remove();
    ghostBubble = null;
  }
}

function addSystemBubble(text) {
  ensureNotEmpty();
  const b = document.createElement('div');
  b.className = 'bubble system';
  b.textContent = text;
  transcriptEl.appendChild(b);
  scrollTranscript();
}

function scrollTranscript() {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ============================================================
// Boot banner (visible until subsystems_ready)
// ============================================================
const bootBanner = $('#boot-banner');
function renderBootBanner(boot, ready) {
  if (!boot) return;
  if (ready) { hideBootBanner(); return; }
  bootBanner.hidden = false;
  for (const [stage, status] of Object.entries(boot)) {
    updateBootPill(stage, status);
  }
}
function updateBootPill(stage, status, detail) {
  const pill = bootBanner.querySelector(`.boot-pill[data-stage="${stage}"]`);
  if (!pill) return;
  pill.dataset.status = status;
  pill.title = detail || status;
  bootBanner.hidden = false;
  // Hide the banner once everything is ready or failed (no more pending).
  const anyPending = Array.from(bootBanner.querySelectorAll('.boot-pill'))
    .some(p => p.dataset.status === 'loading' || p.dataset.status === 'pending' || !p.dataset.status);
  if (!anyPending) hideBootBanner();
}
function hideBootBanner() {
  bootBanner.hidden = true;
}

// ============================================================
// Status rail
// ============================================================
function renderStatus() {
  const h = state.health;
  if (!h) return;
  const grid = $('#status-modules');
  grid.innerHTML = '';
  const order = ['audio', 'stt', 'voice', 'brain', 'tracker'];
  for (const name of order) {
    const m = h.modules?.[name];
    if (!m) continue;
    const cell = document.createElement('div');
    const ok = m.alive !== false && (m.alive !== undefined);
    cell.className = `status-mod ${ok ? 'ok' : 'bad'}`;
    cell.dataset.testid = `status-${name}`;
    cell.innerHTML = `
      <div class="name"><span class="pip"></span>${name}</div>
      <div class="val">${moduleSummary(name, m)}</div>
    `;
    grid.appendChild(cell);
  }
  $('[data-testid="latency-stt"]').textContent = h.latency.stt_ms ? h.latency.stt_ms + ' ms' : '—';
  $('[data-testid="latency-llm"]').textContent = h.latency.llm_ms ? h.latency.llm_ms + ' ms' : '—';
  $('[data-testid="latency-tts"]').textContent = h.latency.tts_ms ? h.latency.tts_ms + ' ms' : '—';
  $('[data-testid="uptime"]').textContent = formatUptime(h.uptime);
}

function moduleSummary(name, m) {
  switch (name) {
    case 'audio': return `dev ${m.device ?? '?'} · ${m.api ?? ''}`;
    case 'stt': return `${m.model} · ${m.on_cpu ? 'CPU' : 'GPU'}`;
    case 'voice': return m.alive ? m.voice : 'offline';
    case 'brain': return m.muted ? 'muted' : (m.in_window ? 'in window' : 'idle');
    case 'tracker': return m.face_present ? 'face present' : (m.alive ? 'watching' : 'offline');
    default: return '';
  }
}

function formatUptime(s) {
  s = Math.floor(s || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

// Refresh /api/health every 4s as a backstop
async function pollHealth() {
  try {
    const r = await fetch('/api/health');
    state.health = await r.json();
    renderStatus();
  } catch {}
}
setInterval(pollHealth, 4000);
pollHealth();

// ============================================================
// VU meter
// ============================================================
const vuCanvas = $('#vu');
const vuCtx = vuCanvas.getContext('2d');
const vuBuffer = new Array(120).fill(0);
let vuThreshold = 0.07, vuOnset = 0.14;

function drawVu(rms, threshold, onset) {
  if (threshold) vuThreshold = threshold;
  if (onset) vuOnset = onset;
  vuBuffer.shift();
  vuBuffer.push(rms);
  const r = vuCanvas.getBoundingClientRect();
  vuCanvas.width = r.width * 2;
  vuCanvas.height = r.height * 2;
  vuCtx.setTransform(2, 0, 0, 2, 0, 0);
  vuCtx.clearRect(0, 0, r.width, r.height);

  // Threshold lines (silence + onset). Both are on a 0..0.4 scale.
  const yScale = (v) => r.height - Math.min(1, v / 0.4) * r.height;
  vuCtx.strokeStyle = 'rgba(217,122,122,0.4)';
  vuCtx.setLineDash([4, 4]);
  vuCtx.beginPath();
  vuCtx.moveTo(0, yScale(vuOnset));
  vuCtx.lineTo(r.width, yScale(vuOnset));
  vuCtx.stroke();

  vuCtx.strokeStyle = 'rgba(110,201,176,0.4)';
  vuCtx.beginPath();
  vuCtx.moveTo(0, yScale(vuThreshold));
  vuCtx.lineTo(r.width, yScale(vuThreshold));
  vuCtx.stroke();
  vuCtx.setLineDash([]);

  // Bars
  const w = r.width / vuBuffer.length;
  for (let i = 0; i < vuBuffer.length; i++) {
    const v = vuBuffer[i];
    const h = Math.min(1, v / 0.4) * r.height;
    const above = v > vuOnset;
    const neutral = v > vuThreshold && !above;
    vuCtx.fillStyle = above ? '#e0b85a' : neutral ? '#b6b09a' : '#444a70';
    vuCtx.fillRect(i * w, r.height - h, w - 0.5, h);
  }
}

// ============================================================
// Camera — poll a single-frame endpoint so the preview gracefully
// survives the tracker still initializing. Refreshes ~6 fps.
// ============================================================
function attachCamera() {
  const img = $('#cam');
  let lastUrl = null;
  let inflight = false;

  async function tick() {
    if (inflight) return;
    inflight = true;
    try {
      const r = await fetch(`/camera.jpg?t=${Date.now()}`, { cache: 'no-store' });
      if (!r.ok) throw new Error('http ' + r.status);
      const blob = await r.blob();
      if (blob.size < 100) return;  // placeholder, ignore
      const url = URL.createObjectURL(blob);
      img.src = url;
      img.style.opacity = 1;
      if (lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl = url;
    } catch (e) {
      img.style.opacity = 0.3;
    } finally {
      inflight = false;
    }
  }
  setInterval(tick, 160);
  tick();
}
attachCamera();

// ============================================================
// Say form
// ============================================================
function sendToBrain(text) {
  if (!text || !text.trim()) return;
  text = text.trim();
  addOptimisticUserBubble(text);
  fetch('/api/say', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({text}),
  }).catch(err => {
    addSystemBubble(`✗ network error: ${err.message}`);
  });
}

$('#say-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const input = $('#say-input');
  const text = input.value;
  input.value = '';
  sendToBrain(text);
});

// ============================================================
// Orb interactions
// ============================================================
$('#orb').addEventListener('click', () => {
  const muted = state.health?.modules?.brain?.muted;
  fetch(`/api/command/${muted ? 'unmute' : 'mute'}`, { method: 'POST' });
});

// ============================================================
// Push-to-talk (Space) and global keys
// ============================================================
let spaceDown = false;
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea')) return;
  if (e.key === ' ' && !spaceDown) {
    spaceDown = true;
    e.preventDefault();
    fetch('/api/command/unmute', { method: 'POST' });
  } else if (e.key.toLowerCase() === 'm') {
    const muted = state.health?.modules?.brain?.muted;
    fetch(`/api/command/${muted ? 'unmute' : 'mute'}`, { method: 'POST' });
  } else if (e.key === 'Escape') {
    if (palette.classList.contains('open')) closePalette();
    else if (drawer.classList.contains('open')) closeDrawer();
    else if (spellbook.classList.contains('open')) closeSpellbook();
    else fetch('/api/command/skip_tts', { method: 'POST' });
  } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    openPalette();
  } else if ((e.metaKey || e.ctrlKey) && e.key === ',') {
    e.preventDefault();
    openDrawer();
  }
});
document.addEventListener('keyup', (e) => {
  if (e.key === ' ') spaceDown = false;
});

// ============================================================
// Settings drawer
// ============================================================
const drawer = $('#drawer');
const drawerScrim = $('#drawer-scrim');
const drawerBody = $('#drawer-body');

function openDrawer() {
  drawer.classList.add('open');
  drawerScrim.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  loadSettings().then(() => renderDrawerTab(currentTab));
}
function closeDrawer() {
  drawer.classList.remove('open');
  drawerScrim.classList.remove('open');
  drawer.setAttribute('aria-hidden', 'true');
}
$('#btn-settings').addEventListener('click', openDrawer);
$('#drawer-close').addEventListener('click', closeDrawer);
drawerScrim.addEventListener('click', () => { closeDrawer(); closeSpellbook(); });

let currentTab = 'hearing';
$$('.drawer-tabs .tab').forEach(t => {
  t.addEventListener('click', () => {
    $$('.drawer-tabs .tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    currentTab = t.dataset.tab;
    renderDrawerTab(currentTab);
  });
});

async function loadSettings() {
  const [s, v, d] = await Promise.all([
    fetch('/api/settings').then(r => r.json()),
    fetch('/api/voices').then(r => r.json()),
    fetch('/api/devices').then(r => r.json()),
  ]);
  state.settings = s;
  state.voices = v;
  state.devices = d;
}

async function patchSetting(key, value) {
  const r = await fetch('/api/settings', {
    method: 'PATCH',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ [key]: value }),
  });
  return await r.json();
}

function fld(label, body, tag, help) {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  const top = document.createElement('div');
  top.className = 'field-row';
  const lbl = document.createElement('label'); lbl.textContent = label;
  top.appendChild(lbl);
  if (tag) {
    const t = document.createElement('span');
    t.className = `tag-pill ${tag}`;
    t.textContent = tag;
    top.appendChild(t);
  }
  wrap.appendChild(top);
  wrap.appendChild(body);
  if (help) {
    const h = document.createElement('div'); h.className = 'field-help'; h.textContent = help;
    wrap.appendChild(h);
  }
  return wrap;
}

function input(value, type, onChange, attrs = {}) {
  const el = document.createElement('input');
  el.type = type;
  el.value = value ?? '';
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  el.addEventListener('change', () => onChange(type === 'number' ? Number(el.value) : el.value));
  return el;
}

function range(value, min, max, step, onChange) {
  const wrap = document.createElement('div');
  wrap.style.display = 'flex'; wrap.style.gap = '10px'; wrap.style.alignItems = 'center';
  const r = document.createElement('input');
  r.type = 'range'; r.min = min; r.max = max; r.step = step; r.value = value;
  const out = document.createElement('span'); out.style.fontFamily = 'var(--font-mono)'; out.style.fontSize = '12px'; out.style.color = 'var(--ink-soft)'; out.style.width = '60px';
  out.textContent = value;
  r.addEventListener('input', () => out.textContent = Number(r.value).toFixed(step < 1 ? 2 : 0));
  r.addEventListener('change', () => onChange(Number(r.value)));
  wrap.appendChild(r); wrap.appendChild(out);
  return wrap;
}

function select(value, options, onChange) {
  const el = document.createElement('select');
  for (const opt of options) {
    const o = document.createElement('option');
    o.value = opt.value; o.textContent = opt.label;
    if (String(opt.value) === String(value)) o.selected = true;
    el.appendChild(o);
  }
  el.addEventListener('change', () => onChange(el.value));
  return el;
}

function chipList(values, onChange) {
  const wrap = document.createElement('div');
  wrap.className = 'chip-list';
  const render = () => {
    wrap.innerHTML = '';
    values.forEach((v, i) => {
      const c = document.createElement('span');
      c.className = 'chip-x';
      c.innerHTML = `<span></span><button aria-label="remove">×</button>`;
      c.querySelector('span').textContent = v;
      c.querySelector('button').addEventListener('click', () => {
        values.splice(i, 1); render(); onChange([...values]);
      });
      wrap.appendChild(c);
    });
    const inp = document.createElement('input');
    inp.placeholder = '+ add…';
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && inp.value.trim()) {
        values.push(inp.value.trim()); inp.value = '';
        render(); onChange([...values]);
      }
    });
    wrap.appendChild(inp);
  };
  render();
  return wrap;
}

function getVal(key) {
  return state.settings?.[key]?.value;
}
function tagOf(key) {
  return state.settings?.[key]?.tag;
}

function renderDrawerTab(tab) {
  drawerBody.innerHTML = '';
  if (!state.settings) return;
  const sub = subsystemFor(tab);

  if (tab === 'hearing') {
    drawerBody.appendChild(fld('Microphone',
      select(state.devices?.current_mic ?? '',
        (state.devices?.inputs || []).map(d => ({ value: d.index, label: `[${d.index}] ${d.name} (${d.api})` })),
        v => patchSetting('PIXY_MIC_DEVICE', v === '' ? null : Number(v)).then(showRestart)),
      tagOf('PIXY_MIC_DEVICE'),
      'Choose the input device. Restart the audio subsystem to apply.'));

    drawerBody.appendChild(fld('Energy threshold',
      range(getVal('ENERGY_THRESHOLD'), 0.01, 0.3, 0.01, v => patchSetting('ENERGY_THRESHOLD', v)),
      tagOf('ENERGY_THRESHOLD'),
      'RMS at which a frame is considered "voiced". Onset = 2× this. Watch the VU meter to tune.'));

    drawerBody.appendChild(fld('Silence timeout (s)',
      input(getVal('SILENCE_TIMEOUT'), 'number', v => patchSetting('SILENCE_TIMEOUT', v), { step: 0.1, min: 0.3, max: 5 }),
      tagOf('SILENCE_TIMEOUT')));

    drawerBody.appendChild(fld('Min utterance (s)',
      input(getVal('MIN_UTTERANCE_LENGTH'), 'number', v => patchSetting('MIN_UTTERANCE_LENGTH', v), { step: 0.1, min: 0.1 }),
      tagOf('MIN_UTTERANCE_LENGTH')));

    drawerBody.appendChild(fld('Max utterance (s)',
      input(getVal('MAX_UTTERANCE_LENGTH'), 'number', v => patchSetting('MAX_UTTERANCE_LENGTH', v), { step: 1, min: 5 }),
      tagOf('MAX_UTTERANCE_LENGTH')));

    drawerBody.appendChild(fld('Whisper model',
      select(getVal('WHISPER_MODEL'),
        ['tiny','base','small','medium'].map(m => ({ value: m, label: m })),
        v => patchSetting('WHISPER_MODEL', v).then(showRestart)),
      tagOf('WHISPER_MODEL')));

    drawerBody.appendChild(fld('Whisper device',
      select(getVal('WHISPER_DEVICE'),
        [{value:'cuda',label:'CUDA (GPU)'},{value:'cpu',label:'CPU'}],
        v => patchSetting('WHISPER_DEVICE', v).then(showRestart)),
      tagOf('WHISPER_DEVICE')));

    drawerBody.appendChild(restartButton('audio'));
    drawerBody.appendChild(restartButton('stt'));
  }

  else if (tab === 'voice') {
    const grid = document.createElement('div');
    grid.className = 'voice-grid';
    const current = getVal('KOKORO_VOICE');
    (state.voices?.voices || []).forEach(v => {
      const cell = document.createElement('div');
      cell.className = `voice-cell ${v.id === current ? 'selected' : ''}`;
      cell.dataset.testid = `voice-${v.id}`;
      cell.innerHTML = `
        <div>
          <div style="font-weight:500">${v.label}</div>
          <div style="font-size:10px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:0.1em">${v.gender} · ${v.accent}</div>
        </div>
        <button class="play" aria-label="preview">▶</button>`;
      cell.querySelector('.play').addEventListener('click', (e) => {
        e.stopPropagation();
        previewVoice(v.id);
      });
      cell.addEventListener('click', () => {
        patchSetting('KOKORO_VOICE', v.id).then(() => {
          $$('.voice-cell').forEach(c => c.classList.remove('selected'));
          cell.classList.add('selected');
        });
      });
      grid.appendChild(cell);
    });
    drawerBody.appendChild(fld('Kokoro voice', grid, 'hot', 'Click to switch. ▶ to preview.'));

    drawerBody.appendChild(fld('Speed',
      range(getVal('KOKORO_SPEED'), 0.5, 1.6, 0.05, v => patchSetting('KOKORO_SPEED', v)),
      tagOf('KOKORO_SPEED')));

    drawerBody.appendChild(fld('Speaker',
      select(getVal('SPEAKER_DEVICE') ?? '',
        [{ value: '', label: 'System default' }, ...(state.devices?.outputs || []).map(d => ({ value: d.index, label: `[${d.index}] ${d.name}` }))],
        v => patchSetting('SPEAKER_DEVICE', v === '' ? null : Number(v))),
      tagOf('SPEAKER_DEVICE')));
  }

  else if (tab === 'sight') {
    drawerBody.appendChild(fld('Camera',
      select(getVal('CAMERA_INDEX') ?? '',
        [{ value:'', label:'auto-detect' }, {value:0,label:'index 0'},{value:1,label:'index 1'},{value:2,label:'index 2'},{value:3,label:'index 3'}],
        v => patchSetting('CAMERA_INDEX', v === '' ? null : Number(v)).then(showRestart)),
      tagOf('CAMERA_INDEX')));

    drawerBody.appendChild(fld('Resolution width',
      input(getVal('CAMERA_WIDTH'), 'number', v => patchSetting('CAMERA_WIDTH', v).then(showRestart)),
      tagOf('CAMERA_WIDTH')));
    drawerBody.appendChild(fld('Resolution height',
      input(getVal('CAMERA_HEIGHT'), 'number', v => patchSetting('CAMERA_HEIGHT', v).then(showRestart)),
      tagOf('CAMERA_HEIGHT')));
    drawerBody.appendChild(fld('FPS',
      input(getVal('CAMERA_FPS'), 'number', v => patchSetting('CAMERA_FPS', v).then(showRestart)),
      tagOf('CAMERA_FPS')));

    drawerBody.appendChild(fld('Face confidence',
      range(getVal('FACE_CONFIDENCE'), 0.3, 0.95, 0.05, v => patchSetting('FACE_CONFIDENCE', v)),
      tagOf('FACE_CONFIDENCE')));

    drawerBody.appendChild(fld('PTZ enabled',
      checkbox(getVal('PTZ_ENABLED'), v => patchSetting('PTZ_ENABLED', v)),
      tagOf('PTZ_ENABLED')));
  }

  else if (tab === 'mind') {
    drawerBody.appendChild(fld('LM Studio URL',
      input(getVal('LLM_URL'), 'text', v => patchSetting('LLM_URL', v)),
      tagOf('LLM_URL')));

    const ta = document.createElement('textarea');
    ta.value = getVal('SYSTEM_PROMPT') || '';
    ta.addEventListener('change', () => patchSetting('SYSTEM_PROMPT', ta.value));
    drawerBody.appendChild(fld('System prompt', ta, tagOf('SYSTEM_PROMPT'),
      'Tell Merlin who he is and how to behave. Applied immediately.'));

    drawerBody.appendChild(fld('Temperature',
      range(getVal('TEMPERATURE'), 0, 1.5, 0.05, v => patchSetting('TEMPERATURE', v)),
      tagOf('TEMPERATURE')));

    drawerBody.appendChild(fld('Max tokens',
      input(getVal('MAX_TOKENS'), 'number', v => patchSetting('MAX_TOKENS', v), { min: 20, max: 1000 }),
      tagOf('MAX_TOKENS')));

    drawerBody.appendChild(fld('Max history exchanges',
      input(getVal('MAX_HISTORY'), 'number', v => patchSetting('MAX_HISTORY', v), { min: 1, max: 50 }),
      tagOf('MAX_HISTORY')));

    drawerBody.appendChild(fld('Conversation window (s)',
      input(getVal('CONVERSATION_WINDOW'), 'number', v => patchSetting('CONVERSATION_WINDOW', v), { min: 5, max: 300 }),
      tagOf('CONVERSATION_WINDOW'),
      'After Merlin replies, you can speak without "Hey Merlin" for this many seconds.'));

    drawerBody.appendChild(fld('Wake words',
      chipList([...(getVal('WAKE_WORDS') || [])], v => patchSetting('WAKE_WORDS', v)),
      tagOf('WAKE_WORDS')));

    drawerBody.appendChild(fld('Mute words',
      chipList([...(getVal('MUTE_WORDS') || [])], v => patchSetting('MUTE_WORDS', v)),
      tagOf('MUTE_WORDS')));

    drawerBody.appendChild(fld('Unmute words',
      chipList([...(getVal('UNMUTE_WORDS') || [])], v => patchSetting('UNMUTE_WORDS', v)),
      tagOf('UNMUTE_WORDS')));

    const clearBtn = document.createElement('button');
    clearBtn.className = 'ghost-btn';
    clearBtn.textContent = 'Clear conversation history';
    clearBtn.addEventListener('click', () => fetch('/api/command/clear_history', { method: 'POST' }));
    drawerBody.appendChild(clearBtn);
  }

  else if (tab === 'appearance') {
    const intensity = parseFloat(localStorage.getItem('orb_intensity') || '1');
    drawerBody.appendChild(fld('Orb intensity',
      range(intensity, 0.2, 1.5, 0.05, v => { localStorage.setItem('orb_intensity', v); orb.setIntensity(v); }),
      'hot', 'Animation amplitude. Lower for less motion.'));

    drawerBody.appendChild(fld('Reduced motion',
      checkbox(localStorage.getItem('reduced_motion') === '1', v => {
        localStorage.setItem('reduced_motion', v ? '1' : '0');
        document.documentElement.style.setProperty('--reduced', v ? '1' : '0');
        document.body.classList.toggle('reduced-motion', v);
      }),
      'hot', 'Pause the starfield drift.'));
  }
}

function checkbox(value, onChange) {
  const wrap = document.createElement('label');
  wrap.style.display = 'inline-flex'; wrap.style.gap = '8px'; wrap.style.alignItems = 'center';
  const c = document.createElement('input');
  c.type = 'checkbox'; c.checked = !!value;
  c.addEventListener('change', () => onChange(c.checked));
  wrap.appendChild(c);
  const span = document.createElement('span'); span.textContent = value ? 'on' : 'off';
  c.addEventListener('change', () => span.textContent = c.checked ? 'on' : 'off');
  wrap.appendChild(span);
  return wrap;
}

function subsystemFor(tab) {
  return ({hearing:'audio', voice:'voice', sight:'tracker', mind:'brain', appearance:null})[tab];
}

function restartButton(subsystem) {
  const wrap = document.createElement('div');
  wrap.className = 'subsystem-actions';
  const b = document.createElement('button');
  b.className = 'ghost-btn';
  b.textContent = `Restart ${subsystem}`;
  b.title = `Stops and re-initializes the ${subsystem} subsystem so restart-required settings take effect.`;
  b.addEventListener('click', () => {
    addSystemBubble(`restart ${subsystem} not yet wired — quit & relaunch Merlin to apply`);
  });
  wrap.appendChild(b);
  return wrap;
}

function showRestart(result) {
  const restarting = Object.values(result).find(r => r.status === 'restart_required');
  if (restarting) addSystemBubble(`restart ${restarting.subsystem} to apply`);
}

async function previewVoice(voiceId) {
  try {
    const r = await fetch('/api/voices/preview', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ voice: voiceId, text: 'Hello. I am Merlin.' }),
    });
    const blob = await r.blob();
    const audio = $('#preview-audio');
    audio.src = URL.createObjectURL(blob);
    audio.play();
  } catch (e) { console.error(e); }
}

// ============================================================
// Command palette
// ============================================================
const palette = $('#palette');
const paletteInput = $('#palette-input');
const paletteList = $('#palette-list');

const COMMANDS = [
  { id: 'mute', label: 'Mute Merlin', hint: 'M', run: () => fetch('/api/command/mute', {method:'POST'}) },
  { id: 'unmute', label: 'Unmute Merlin', hint: 'M', run: () => fetch('/api/command/unmute', {method:'POST'}) },
  { id: 'clear', label: 'Clear conversation history', run: () => fetch('/api/command/clear_history', {method:'POST'}) },
  { id: 'skip', label: 'Skip current TTS', hint: 'Esc', run: () => fetch('/api/command/skip_tts', {method:'POST'}) },
  { id: 'settings', label: 'Open settings', hint: 'Ctrl+,', run: () => { closePalette(); openDrawer(); } },
  { id: 'spellbook', label: 'Open spellbook', run: () => { closePalette(); openSpellbook(); } },
  { id: 'export', label: 'Export current session as markdown', run: exportSession },
];

function openPalette() {
  palette.classList.add('open');
  paletteInput.value = '';
  renderPalette('');
  paletteInput.focus();
}
function closePalette() {
  palette.classList.remove('open');
}
$('#btn-palette').addEventListener('click', openPalette);
palette.addEventListener('click', (e) => { if (e.target === palette) closePalette(); });

let paletteSelected = 0;
function renderPalette(q) {
  const list = COMMANDS.filter(c => c.label.toLowerCase().includes(q.toLowerCase()));
  paletteList.innerHTML = '';
  list.forEach((c, i) => {
    const li = document.createElement('li');
    li.role = 'option';
    if (i === paletteSelected) li.classList.add('selected');
    li.innerHTML = `<span>${c.label}</span>${c.hint ? `<span class="hint">${c.hint}</span>` : ''}`;
    li.addEventListener('click', () => { c.run(); closePalette(); });
    paletteList.appendChild(li);
  });
  paletteList._items = list;
}
paletteInput.addEventListener('input', () => { paletteSelected = 0; renderPalette(paletteInput.value); });
paletteInput.addEventListener('keydown', (e) => {
  const items = paletteList._items || [];
  if (e.key === 'ArrowDown') { paletteSelected = (paletteSelected + 1) % Math.max(1, items.length); renderPalette(paletteInput.value); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { paletteSelected = (paletteSelected - 1 + items.length) % Math.max(1, items.length); renderPalette(paletteInput.value); e.preventDefault(); }
  else if (e.key === 'Enter') { items[paletteSelected]?.run(); closePalette(); }
});

// ============================================================
// Spellbook
// ============================================================
const spellbook = $('#spellbook');
const spellList = $('#spell-list');
const spellbookChips = $('#spellbook-chips');
let spells = JSON.parse(localStorage.getItem('spells') || '["What time is it?","Summarize the last hour.","What\'s on my plate today?","Tell me a story."]');

function saveSpells() { localStorage.setItem('spells', JSON.stringify(spells)); renderSpells(); }
function renderSpells() {
  spellList.innerHTML = '';
  spells.forEach((s, i) => {
    const li = document.createElement('li');
    li.dataset.testid = `spell-${i}`;
    li.innerHTML = `<span></span><button class="del" aria-label="delete">×</button>`;
    li.querySelector('span').textContent = s;
    li.addEventListener('click', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      sendToBrain(s);
      closeSpellbook();
    });
    li.querySelector('.del').addEventListener('click', (e) => {
      e.stopPropagation(); spells.splice(i, 1); saveSpells();
    });
    spellList.appendChild(li);
  });
  // mini chips on the right rail
  spellbookChips.innerHTML = '';
  spells.slice(0, 6).forEach(s => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = s.length > 22 ? s.slice(0, 21) + '…' : s;
    chip.title = s;
    chip.addEventListener('click', () => sendToBrain(s));
    spellbookChips.appendChild(chip);
  });
}
$('#spell-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const inp = $('#spell-text');
  if (inp.value.trim()) { spells.push(inp.value.trim()); inp.value = ''; saveSpells(); }
});
function openSpellbook() { spellbook.classList.add('open'); drawerScrim.classList.add('open'); }
function closeSpellbook() { spellbook.classList.remove('open'); drawerScrim.classList.remove('open'); }
$('#btn-spellbook').addEventListener('click', openSpellbook);
$('#spellbook-close').addEventListener('click', closeSpellbook);

renderSpells();

// ============================================================
// Session picker + history search
// ============================================================
async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    const data = await r.json();
    const sel = $('#session-picker');
    sel.innerHTML = '<option value="">Current session</option>';
    data.sessions.forEach(s => {
      const o = document.createElement('option');
      o.value = s.id;
      const date = new Date(s.started_at * 1000);
      o.textContent = `${date.toLocaleString()} (${s.msg_count})`;
      sel.appendChild(o);
    });
  } catch {}
}
$('#session-picker').addEventListener('change', async (e) => {
  const id = e.target.value;
  if (!id) { transcriptEl.innerHTML = '<div class="empty">live session.</div>'; return; }
  const r = await fetch(`/api/history?session=${id}`);
  const data = await r.json();
  transcriptEl.innerHTML = '';
  data.messages.forEach(m => addBubble(m.role === 'assistant' ? 'merlin' : 'you', m.text, { latency_ms: m.latency_ms }));
  addSystemBubble('— end of session —');
});
loadSessions();

// History full-text search
let searchTimer;
$('#history-search').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  if (!q) { return; }
  searchTimer = setTimeout(async () => {
    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await r.json();
    transcriptEl.innerHTML = '';
    if (!data.results.length) addSystemBubble('no matches');
    data.results.forEach(m => {
      addBubble(m.role === 'assistant' ? 'merlin' : 'you', m.text);
    });
    addSystemBubble(`— ${data.results.length} matches —`);
  }, 250);
});

// ============================================================
// Export
// ============================================================
async function exportSession() {
  const r = await fetch('/api/history?limit=1000');
  const data = await r.json();
  const md = data.messages.map(m =>
    `**${m.role === 'assistant' ? 'Merlin' : 'You'}** _(${new Date(m.ts*1000).toLocaleString()})_:\n${m.text}\n`
  ).join('\n');
  const blob = new Blob([md], {type:'text/markdown'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `merlin-${new Date().toISOString().slice(0,16).replace(':','-')}.md`;
  a.click();
}

// ============================================================
// Initial intensity & motion
// ============================================================
const storedIntensity = parseFloat(localStorage.getItem('orb_intensity') || '1');
orb.setIntensity(storedIntensity);
if (localStorage.getItem('reduced_motion') === '1') {
  document.body.classList.add('reduced-motion');
}

// ============================================================
// Go!
// ============================================================
connectWs();
setState('idle');
