/**
 * Port of sim/ball.py. Kept deliberately line-for-line with the Python so the
 * live simulation on /try reproduces the training ground truth exactly rather
 * than merely looking similar.
 *
 * Velocity Verlet is exact for a constant force, so free flight conserves
 * energy to machine precision; contact is resolved at the exact floor-crossing
 * time (the positive root of the flight parabola) and flips velocity by the
 * coefficient of restitution.
 */

export interface PhysicsConfig {
  g: number;
  dt: number;
  restitution: number;
  n_steps: number;
  q0_range?: [number, number];
  v0_range?: [number, number];
}

export const DEFAULT_CONFIG: PhysicsConfig = {
  g: 9.81,
  dt: 0.01,
  restitution: 0.8,
  n_steps: 200,
};

const MAX_BOUNCES = 100;

export function energy(q: number, p: number, m: number, g: number): number {
  return (p * p) / (2 * m) + m * g * q;
}

/** One velocity-Verlet step with event-resolved contact. */
export function step(
  q: number,
  p: number,
  m: number,
  dt: number,
  g: number,
  e: number,
): { q: number; p: number; bounces: number } {
  let v = p / m;
  let remaining = dt;
  let bounces = 0;

  for (;;) {
    const qNext = q + v * remaining - 0.5 * g * remaining * remaining;
    // Free flight is a downward parabola, so its minimum over the interval is
    // at the right endpoint; qNext >= 0 rules out any crossing in between.
    if (qNext >= 0) {
      return { q: qNext, p: m * (v - g * remaining), bounces };
    }

    // Positive root of q + v s - g s^2 / 2 = 0.
    const tHit = (v + Math.sqrt(v * v + 2 * g * q)) / g;
    v = -e * (v - g * tHit);
    q = 0;
    remaining -= tHit;
    bounces += 1;

    // e -> 0 gives infinitely many bounces in finite time.
    if (bounces > MAX_BOUNCES) return { q: 0, p: 0, bounces };
  }
}

export interface Trace {
  t: number[];
  q: number[];
  p: number[];
  energy: number[];
  /** Indices at which one or more contacts occurred. */
  bounceAt: number[];
}

export function simulate(
  q0: number,
  p0: number,
  m: number,
  cfg: PhysicsConfig,
): Trace {
  const n = cfg.n_steps;
  const t = new Array<number>(n + 1);
  const q = new Array<number>(n + 1);
  const p = new Array<number>(n + 1);
  const e = new Array<number>(n + 1);
  const bounceAt: number[] = [];

  q[0] = q0;
  p[0] = p0;
  t[0] = 0;
  e[0] = energy(q0, p0, m, cfg.g);

  for (let i = 0; i < n; i++) {
    const next = step(q[i], p[i], m, cfg.dt, cfg.g, cfg.restitution);
    q[i + 1] = next.q;
    p[i + 1] = next.p;
    t[i + 1] = (i + 1) * cfg.dt;
    e[i + 1] = energy(next.q, next.p, m, cfg.g);
    if (next.bounces > 0) bounceAt.push(i + 1);
  }

  return { t, q, p, energy: e, bounceAt };
}
