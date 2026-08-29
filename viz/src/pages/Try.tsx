import { useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_CONFIG, simulate, type Trace } from "../sim/ball";

const HORIZON = 1400; // steps; at dt = 0.01 s that is 14 s of simulated time

interface Params {
  q0: number;
  v0: number;
  restitution: number;
  g: number;
  mass: number;
}

const PRESETS: { name: string; hint: string; params: Params }[] = [
  {
    name: "Training regime",
    hint: "The exact setting the models were trained on: e = 0.8.",
    params: { q0: 1.5, v0: 0, restitution: 0.8, g: 9.81, mass: 1 },
  },
  {
    name: "Elastic",
    hint: "e = 1.0. Energy is exactly conserved — the system is genuinely Hamiltonian.",
    params: { q0: 1.5, v0: 0, restitution: 1.0, g: 9.81, mass: 1 },
  },
  {
    name: "Dead stop",
    hint: "e = 0.3. Energy collapses in a few bounces and the ball settles.",
    params: { q0: 2.0, v0: 0, restitution: 0.3, g: 9.81, mass: 1 },
  },
  {
    name: "Thrown down",
    hint: "Launched downward, so the first contact comes fast and hard.",
    params: { q0: 1.0, v0: -4, restitution: 0.85, g: 9.81, mass: 1 },
  },
  {
    name: "Lunar",
    hint: "g = 1.62. Slow, floaty arcs with the same restitution.",
    params: { q0: 2.0, v0: 0, restitution: 0.85, g: 1.62, mass: 1 },
  },
];

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="slider">
      <span className="slider-label">
        {props.label}
        <b>
          {props.value.toFixed(2)} {props.unit}
        </b>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function Try() {
  const [params, setParams] = useState<Params>(PRESETS[0].params);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [frame, setFrame] = useState(0);

  const set = <K extends keyof Params>(k: K, v: Params[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

  // Recomputed only when physics changes; the animation just indexes into it,
  // so dragging a slider never competes with the render loop.
  const trace: Trace = useMemo(
    () =>
      simulate(params.q0, params.v0 * params.mass, params.mass, {
        ...DEFAULT_CONFIG,
        g: params.g,
        restitution: params.restitution,
        n_steps: HORIZON,
      }),
    [params],
  );

  useEffect(() => setFrame(0), [trace]);

  const ballRef = useRef<HTMLCanvasElement>(null);
  const plotRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef(0);
  frameRef.current = frame;

  // Animation clock. dt is simulated seconds per step, so advancing
  // speed/dt steps per second plays back at `speed` x real time.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    // The rAF timestamp is the start of the frame, which can predate a
    // performance.now() taken here, so seed it on the first tick instead of
    // now. Clamping below zero as well keeps a bad delta from ever rewinding
    // the index past the start of the trace.
    let last: number | null = null;
    let acc = frameRef.current;

    const tick = (now: number) => {
      const elapsed = last === null ? 0 : Math.min(Math.max(now - last, 0) / 1000, 0.1);
      last = now;
      acc += (elapsed * speed) / DEFAULT_CONFIG.dt;
      if (acc >= HORIZON) acc = 0;
      setFrame(acc);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, trace]);

  // Clamped both ends: idx indexes the trace directly, so it must stay in range
  // no matter what the clock or the scrub control produces.
  const idx = Math.max(0, Math.min(Math.floor(frame) || 0, HORIZON));
  const qMax = useMemo(() => Math.max(...trace.q, 0.5) * 1.12, [trace]);
  const eMax = useMemo(() => Math.max(...trace.energy, 1e-6), [trace]);

  // Ball canvas.
  useEffect(() => {
    const cv = ballRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
      cv.width = w * dpr;
      cv.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const padTop = 26;
    const floorY = h - 34;
    const toY = (q: number) => floorY - (q / qMax) * (floorY - padTop);
    const cx = w / 2;

    // Floor.
    ctx.strokeStyle = "#3a4256";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(24, floorY);
    ctx.lineTo(w - 24, floorY);
    ctx.stroke();
    ctx.fillStyle = "rgba(120,132,160,0.10)";
    ctx.fillRect(24, floorY, w - 48, h - floorY);

    // Height gridlines.
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "left";
    for (let k = 0; k <= 4; k++) {
      const q = (qMax * k) / 4;
      const y = toY(q);
      ctx.strokeStyle = "rgba(120,132,160,0.13)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(24, y);
      ctx.lineTo(w - 24, y);
      ctx.stroke();
      ctx.fillStyle = "#6b7590";
      ctx.fillText(`${q.toFixed(1)} m`, 28, y - 4);
    }

    // Motion trail. Short enough that a fast ball reads as a comet rather than
    // a solid streak down the whole column.
    const trail = 26;
    for (let k = Math.max(0, idx - trail); k < idx; k++) {
      const a = (k - (idx - trail)) / trail;
      ctx.fillStyle = `rgba(233,196,106,${0.20 * a * a})`;
      ctx.beginPath();
      ctx.arc(cx, toY(trace.q[k]), 13 * (0.3 + 0.7 * a), 0, Math.PI * 2);
      ctx.fill();
    }

    // Ball, squashed slightly when it is near the floor.
    const q = trace.q[idx];
    const y = toY(q);
    const squash = Math.max(0, 1 - q / (qMax * 0.05));
    ctx.save();
    ctx.translate(cx, Math.min(y, floorY - 13 * (1 - 0.35 * squash)));
    ctx.scale(1 + 0.3 * squash, 1 - 0.3 * squash);
    const grad = ctx.createRadialGradient(-4, -5, 2, 0, 0, 15);
    grad.addColorStop(0, "#ffe9a8");
    grad.addColorStop(1, "#e9a23b");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }, [idx, trace, qMax]);

  // Energy + height trace canvas.
  useEffect(() => {
    const cv = plotRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
      cv.width = w * dpr;
      cv.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = 30;
    const x = (i: number) => pad + (i / HORIZON) * (w - pad - 12);
    const yE = (v: number) => h - 22 - (v / eMax) * (h - 44);
    const yQ = (v: number) => h - 22 - (v / qMax) * (h - 44);

    ctx.strokeStyle = "rgba(120,132,160,0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - 22);
    ctx.lineTo(w - 12, h - 22);
    ctx.stroke();

    const line = (
      fn: (i: number) => number,
      color: string,
      width: number,
      alpha: number,
    ) => {
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (let i = 0; i <= idx; i++) {
        const px = x(i);
        const py = fn(i);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    line((i) => yQ(trace.q[i]), "#e9a23b", 1.2, 0.45);
    line((i) => yE(trace.energy[i]), "#4ade80", 2, 1);

    // Playhead.
    ctx.strokeStyle = "rgba(255,255,255,0.28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(idx), 8);
    ctx.lineTo(x(idx), h - 22);
    ctx.stroke();

    ctx.font = "11px ui-monospace, monospace";
    ctx.fillStyle = "#4ade80";
    ctx.textAlign = "left";
    ctx.fillText("energy", pad + 2, 16);
    ctx.fillStyle = "#e9a23b";
    ctx.fillText("height", pad + 56, 16);
    ctx.fillStyle = "#6b7590";
    ctx.textAlign = "right";
    ctx.fillText(`${(HORIZON * DEFAULT_CONFIG.dt).toFixed(0)} s`, w - 12, h - 8);
    ctx.textAlign = "left";
    ctx.fillText("0 s", pad, h - 8);
  }, [idx, trace, eMax, qMax]);

  const bounces = trace.bounceAt.filter((b) => b <= idx).length;
  const e0 = trace.energy[0];
  const now = trace.energy[idx];
  const retained = e0 > 0 ? (now / e0) * 100 : 0;
  const elastic = params.restitution >= 0.999;

  return (
    <div className="page">
      <section className="page-head">
        <h1>Try it</h1>
        <p className="lede">
          This runs the real simulator, not a recording. The integrator below is
          a line-for-line TypeScript port of <code>sim/ball.py</code> — velocity
          Verlet with contact resolved at the exact floor-crossing time — and it
          is verified against the Python ground truth on every exported
          trajectory.
        </p>
      </section>

      <div className="try-grid">
        <div className="try-stage">
          <canvas ref={ballRef} className="ball-canvas" />
          <div className="readouts">
            <div className="readout">
              <span>Height</span>
              <b>{trace.q[idx].toFixed(3)} m</b>
            </div>
            <div className="readout">
              <span>Momentum</span>
              <b>{trace.p[idx].toFixed(3)}</b>
            </div>
            <div className="readout">
              <span>Energy</span>
              <b>{now.toFixed(3)} J</b>
            </div>
            <div className="readout">
              <span>Bounces</span>
              <b>{bounces}</b>
            </div>
            <div className="readout">
              <span>Energy left</span>
              <b>{retained.toFixed(1)}%</b>
            </div>
            <div className="readout">
              <span>Time</span>
              <b>{(idx * DEFAULT_CONFIG.dt).toFixed(2)} s</b>
            </div>
          </div>
          <canvas ref={plotRef} className="trace-canvas" />
        </div>

        <aside className="try-panel">
          <div className="panel-block">
            <h3>Playback</h3>
            <div className="row">
              <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
                {playing ? "Pause" : "Play"}
              </button>
              <button
                className="btn"
                onClick={() => {
                  setFrame(0);
                  setPlaying(true);
                }}
              >
                Restart
              </button>
            </div>
            <Slider
              label="Speed"
              value={speed}
              min={0.1}
              max={3}
              step={0.1}
              unit="x"
              onChange={setSpeed}
            />
            <label className="slider">
              <span className="slider-label">
                Scrub<b>{(idx * DEFAULT_CONFIG.dt).toFixed(2)} s</b>
              </span>
              <input
                type="range"
                min={0}
                max={HORIZON}
                step={1}
                value={idx}
                onChange={(e) => {
                  setPlaying(false);
                  setFrame(Number(e.target.value));
                }}
              />
            </label>
          </div>

          <div className="panel-block">
            <h3>Initial state</h3>
            <Slider
              label="Drop height"
              value={params.q0}
              min={0.1}
              max={4}
              step={0.05}
              unit="m"
              onChange={(v) => set("q0", v)}
            />
            <Slider
              label="Initial velocity"
              value={params.v0}
              min={-6}
              max={6}
              step={0.1}
              unit="m/s"
              onChange={(v) => set("v0", v)}
            />
          </div>

          <div className="panel-block">
            <h3>Physics</h3>
            <Slider
              label="Restitution e"
              value={params.restitution}
              min={0}
              max={1}
              step={0.01}
              unit=""
              onChange={(v) => set("restitution", v)}
            />
            <Slider
              label="Gravity g"
              value={params.g}
              min={0.5}
              max={20}
              step={0.05}
              unit="m/s²"
              onChange={(v) => set("g", v)}
            />
            <Slider
              label="Mass m"
              value={params.mass}
              min={0.2}
              max={3}
              step={0.05}
              unit="kg"
              onChange={(v) => set("mass", v)}
            />
            <p className={elastic ? "note good" : "note"}>
              {elastic
                ? "e = 1: energy is exactly conserved. This is the only regime where the Hamiltonian assumption holds literally — and the one metric where the HNN beats the Neural ODE."
                : `e = ${params.restitution.toFixed(2)}: each bounce keeps ${(
                    params.restitution ** 2 * 100
                  ).toFixed(0)}% of the kinetic energy. The system is not Hamiltonian across contacts.`}
            </p>
          </div>

          <div className="panel-block">
            <h3>Presets</h3>
            <div className="presets">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  className="preset"
                  title={p.hint}
                  onClick={() => {
                    setParams(p.params);
                    setPlaying(true);
                  }}
                >
                  {p.name}
                </button>
              ))}
            </div>
            <p className="note">
              The models were trained only at m = 1.0 kg, g = 9.81, e = 0.8. Move
              away from that and you are outside the data they ever saw.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
