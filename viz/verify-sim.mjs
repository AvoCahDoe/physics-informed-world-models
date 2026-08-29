/** Checks the TypeScript simulator against the Python ground truth exported in
 *  public/data/, plus energy conservation in the elastic case.
 *
 *  Run: node --experimental-strip-types viz/verify-sim.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
// pathToFileURL: a bare Windows path like C:\... is not a valid ESM specifier.
const { simulate, energy } = await import(
  pathToFileURL(join(HERE, "src/sim/ball.ts")).href
);

const cfg = JSON.parse(readFileSync(join(HERE, "public/data/config.json"), "utf8"));
const results = [];
const check = (name, ok, detail = "") => {
  results.push(ok);
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${detail ? `  (${detail})` : ""}`);
};

// 1. Reproduce every exported ground-truth trajectory from its initial state.
//
// The exported JSON is rounded to 5 decimals, so the initial state the TS sim
// starts from already differs from Python's by up to 5e-6, and bounces amplify
// that. A fixed tolerance would therefore be arbitrary. Instead, re-run from
// the four corners of the rounding box and require the Python truth to lie
// inside the resulting envelope (widened by the 5e-6 rounding on the output
// itself). Passing means the two implementations are bit-compatible up to the
// precision the file actually carries.
const R = 5e-6;
let outside = 0;
let worstQ = 0;
let worstP = 0;
for (let i = 0; i < 6; i++) {
  const traj = JSON.parse(
    readFileSync(join(HERE, `public/data/trajectory_${i}.json`), "utf8"),
  );
  const truth = traj.contact.truth;
  const opts = { ...cfg, restitution: traj.restitution, n_steps: truth.q.length - 1 };

  const corners = [];
  for (const dq of [-R, R]) {
    for (const dp of [-R, R]) {
      corners.push(simulate(truth.q[0] + dq, truth.p[0] + dp, 1.0, opts));
    }
  }
  const nominal = simulate(truth.q[0], truth.p[0], 1.0, opts);

  for (let k = 0; k < truth.q.length; k++) {
    worstQ = Math.max(worstQ, Math.abs(nominal.q[k] - truth.q[k]));
    worstP = Math.max(worstP, Math.abs(nominal.p[k] - truth.p[k]));
    for (const key of ["q", "p"]) {
      const vals = corners.map((c) => c[key][k]).concat(nominal[key][k]);
      const lo = Math.min(...vals) - R;
      const hi = Math.max(...vals) + R;
      if (truth[key][k] < lo || truth[key][k] > hi) outside++;
    }
  }
}
check(
  "TS simulator matches Python ground truth",
  outside === 0,
  `${outside} samples outside rounding envelope; raw max dq=${worstQ.toExponential(2)}, dp=${worstP.toExponential(2)}`,
);

// 2. Elastic case must conserve energy to machine precision.
const el = simulate(1.5, -0.3, 1.0, { ...cfg, restitution: 1.0, n_steps: 4000 });
const spread =
  (Math.max(...el.energy) - Math.min(...el.energy)) / el.energy[0];
check(
  "elastic case conserves energy",
  spread < 1e-10,
  `relative spread ${spread.toExponential(2)} over 4000 steps`,
);

// 3. The ball must never end a step below the floor.
const inel = simulate(2.0, 0.5, 1.0, { ...cfg, restitution: 0.8, n_steps: 2000 });
check("ball never passes through the floor", Math.min(...inel.q) >= 0, `min q=${Math.min(...inel.q).toExponential(2)}`);

// 4. Inelastic energy must be non-increasing.
let monotone = true;
for (let i = 1; i < inel.energy.length; i++) {
  if (inel.energy[i] > inel.energy[i - 1] + 1e-9) monotone = false;
}
check("inelastic energy is non-increasing", monotone,
  `${inel.energy[0].toFixed(2)} -> ${inel.energy.at(-1).toFixed(2)} J`);

// 5. Bounces are actually detected.
check("contacts detected", inel.bounceAt.length > 3, `${inel.bounceAt.length} bounce steps`);

// 6. Energy helper agrees with the analytic form.
check("energy helper correct", Math.abs(energy(2, 3, 1.5, 9.81) - (9 / 3 + 1.5 * 9.81 * 2)) < 1e-12);

const failed = results.filter((r) => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
