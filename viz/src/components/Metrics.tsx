import { useStore } from "../store";
import { COLORS, LABELS, type SeriesKey } from "../types";

const MODELS: SeriesKey[] = ["mlp", "hnn", "node"];

function fmt(v: number | undefined, digits = 2) {
  if (v === undefined || v === null) return "—";
  return v < 0.01 ? v.toExponential(digits) : v.toFixed(digits);
}

export default function Metrics() {
  const metrics = useStore((s) => s.manifest?.metrics);
  if (!metrics) return null;

  return (
    <div className="metrics">
      <h3>Test-set metrics</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Rollout MSE</th>
            <th>Median MSE</th>
            <th>|dE| [J]</th>
            <th>Train</th>
          </tr>
        </thead>
        <tbody>
          {MODELS.map((m) => (
            <tr key={m}>
              <td>
                <span className="dot" style={{ background: COLORS[m] }} />
                {LABELS[m]}
              </td>
              <td>{fmt(metrics[m]?.rollout_mse)}</td>
              <td>{fmt(metrics[m]?.median_traj_mse)}</td>
              <td>{fmt(metrics[m]?.mean_abs_energy_drift)}</td>
              <td>{fmt(metrics[m]?.train_seconds, 0)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        HNN and Neural ODE beat the MLP by ~300x, but not each other: once
        contact is injected explicitly, free flight is linear and leaves the
        Hamiltonian prior little to exploit.
      </p>
    </div>
  );
}
