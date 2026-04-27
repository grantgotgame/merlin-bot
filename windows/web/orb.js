/**
 * The Orb — a canvas-rendered audio-reactive sphere that visualizes
 * Merlin's state. Idle: gentle breathing. Listening: expands with mic
 * RMS. Thinking: swirling particles. Speaking: waveform ring.
 */

export class Orb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);

    this.state = 'idle';     // idle | listening | thinking | speaking | muted
    this.rms = 0;            // 0..1 mic level (smoothed)
    this.rmsTarget = 0;
    this.envelope = [];      // tts envelope from server
    this.envIndex = 0;
    this.envT0 = 0;
    this.envDuration = 0;
    this.particles = [];     // thinking swirl
    this.t = 0;
    this.intensity = 1.0;    // user setting

    this._resize();
    window.addEventListener('resize', () => this._resize());

    this._initParticles();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  setIntensity(v) { this.intensity = Math.max(0.2, Math.min(1.5, v)); }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    if (state === 'thinking') this._initParticles();
  }

  pushRms(rms) {
    this.rmsTarget = Math.min(1, rms * 5);  // scale 0..0.2 → 0..1 ish
  }

  pushEnvelope(envelope, durationSec) {
    this.envelope = envelope || [];
    this.envIndex = 0;
    this.envT0 = performance.now();
    this.envDuration = (durationSec || 1) * 1000;
  }

  _initParticles() {
    this.particles = [];
    const N = 60;
    for (let i = 0; i < N; i++) {
      this.particles.push({
        a: Math.random() * Math.PI * 2,
        r: 0.6 + Math.random() * 0.4,
        speed: 0.5 + Math.random() * 1.0,
        size: 1 + Math.random() * 2,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    this.w = r.width;
    this.h = r.height;
    this.canvas.width = r.width * this.dpr;
    this.canvas.height = r.height * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  _stateColor() {
    switch (this.state) {
      case 'listening': return { core: '#ffd06b', halo: '#e0b85a' };
      case 'thinking':  return { core: '#a89bff', halo: '#8a7af0' };
      case 'speaking':  return { core: '#9af0d0', halo: '#6ec9b0' };
      case 'muted':     return { core: '#6b6750', halo: '#3e3a2a' };
      default:          return { core: '#9aa6e6', halo: '#6e7ab3' };
    }
  }

  _loop(now) {
    this.t += 0.016;
    // Smooth rms
    this.rms += (this.rmsTarget - this.rms) * 0.25;
    this.rmsTarget *= 0.92;
    this._draw(now);
    requestAnimationFrame(this._loop);
  }

  _draw(now) {
    const ctx = this.ctx;
    const w = this.w, h = this.h;
    const cx = w / 2, cy = h / 2;
    const baseR = Math.min(w, h) * 0.28;
    const colors = this._stateColor();

    ctx.clearRect(0, 0, w, h);

    // Breathing modulation
    const breath = 0.5 + 0.5 * Math.sin(this.t * 0.6);
    let radius = baseR * (1 + 0.04 * breath * this.intensity);

    if (this.state === 'listening') {
      radius += baseR * 0.18 * this.rms * this.intensity;
    }

    if (this.state === 'speaking') {
      const env = this._currentEnv();
      radius += baseR * 0.12 * env * this.intensity;
    }

    // Outer halo
    const halo = ctx.createRadialGradient(cx, cy, radius * 0.6, cx, cy, radius * 2.2);
    halo.addColorStop(0, this._hex(colors.halo, 0.35));
    halo.addColorStop(0.5, this._hex(colors.halo, 0.1));
    halo.addColorStop(1, this._hex(colors.halo, 0));
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 2.2, 0, Math.PI * 2);
    ctx.fill();

    // Aura rings
    for (let i = 0; i < 3; i++) {
      const r = radius * (1.2 + i * 0.25 + Math.sin(this.t * 0.7 + i) * 0.04);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = this._hex(colors.halo, 0.15 - i * 0.04);
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Core sphere with shimmer
    const grad = ctx.createRadialGradient(cx - radius * 0.3, cy - radius * 0.4, radius * 0.2, cx, cy, radius);
    grad.addColorStop(0, this._hex(colors.core, 0.95));
    grad.addColorStop(0.5, this._hex(colors.halo, 0.7));
    grad.addColorStop(1, this._hex(colors.halo, 0.15));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    // Inner sigil — subtle rotating geometry
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(this.t * 0.05);
    ctx.strokeStyle = this._hex(colors.core, 0.4);
    ctx.lineWidth = 0.8;
    for (let i = 0; i < 6; i++) {
      ctx.rotate(Math.PI / 3);
      ctx.beginPath();
      ctx.arc(radius * 0.45, 0, radius * 0.5, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();

    // State-specific overlays
    if (this.state === 'thinking') {
      this._drawParticles(cx, cy, radius, colors);
    }
    if (this.state === 'speaking') {
      this._drawWaveformRing(cx, cy, radius, colors);
    }
    if (this.state === 'listening') {
      this._drawListeningRing(cx, cy, radius, colors);
    }
    if (this.state === 'muted') {
      this._drawMutedX(cx, cy, radius, colors);
    }

    // Core highlight (fixed)
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.beginPath();
    ctx.ellipse(cx - radius * 0.35, cy - radius * 0.45, radius * 0.25, radius * 0.12, -0.5, 0, Math.PI * 2);
    ctx.fill();
  }

  _drawListeningRing(cx, cy, r, colors) {
    const ctx = this.ctx;
    const segments = 64;
    ctx.beginPath();
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      const wobble = 1 + 0.04 * Math.sin(a * 5 + this.t * 4) + 0.1 * this.rms;
      const rr = r * 1.08 * wobble;
      const x = cx + Math.cos(a) * rr;
      const y = cy + Math.sin(a) * rr;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = this._hex(colors.core, 0.5 + 0.5 * this.rms);
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  _drawWaveformRing(cx, cy, r, colors) {
    const ctx = this.ctx;
    const env = this.envelope.length ? this.envelope : [0.5, 0.7, 0.4, 0.6, 0.3];
    const N = 96;
    ctx.beginPath();
    const elapsed = (performance.now() - this.envT0) / Math.max(1, this.envDuration);
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2;
      const phase = (i / N + elapsed) % 1;
      const idx = Math.floor(phase * env.length);
      const v = env[idx] || 0;
      const rr = r * (1.08 + 0.18 * v);
      const x = cx + Math.cos(a) * rr;
      const y = cy + Math.sin(a) * rr;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = this._hex(colors.core, 0.7);
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  _drawParticles(cx, cy, r, colors) {
    const ctx = this.ctx;
    for (const p of this.particles) {
      p.a += 0.005 * p.speed;
      const orbit = r * (1.15 + 0.2 * Math.sin(this.t * p.speed + p.phase));
      const x = cx + Math.cos(p.a) * orbit * p.r;
      const y = cy + Math.sin(p.a) * orbit * p.r;
      ctx.fillStyle = this._hex(colors.core, 0.6);
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawMutedX(cx, cy, r, colors) {
    const ctx = this.ctx;
    ctx.strokeStyle = this._hex(colors.core, 0.7);
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    const d = r * 0.4;
    ctx.beginPath();
    ctx.moveTo(cx - d, cy - d);
    ctx.lineTo(cx + d, cy + d);
    ctx.moveTo(cx + d, cy - d);
    ctx.lineTo(cx - d, cy + d);
    ctx.stroke();
  }

  _currentEnv() {
    if (!this.envelope.length || this.envDuration === 0) return 0;
    const elapsed = performance.now() - this.envT0;
    if (elapsed > this.envDuration) return 0;
    const idx = Math.floor((elapsed / this.envDuration) * this.envelope.length);
    return this.envelope[Math.min(idx, this.envelope.length - 1)] || 0;
  }

  _hex(hex, alpha) {
    const m = hex.match(/^#([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i);
    if (!m) return hex;
    const r = parseInt(m[1], 16);
    const g = parseInt(m[2], 16);
    const b = parseInt(m[3], 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
}
