export type SeriesKey = "truth" | "mlp" | "hnn" | "node";

export interface Series {
  q: number[];
  p: number[];
  energy: number[];
}

export interface Trajectory {
  id: number;
  t: number[];
  dt: number;
  restitution: number;
  g: number;
  contact: Record<string, Series>;
  naive: Record<string, Series>;
}

export interface TrajectoryMeta {
  id: number;
  file: string;
  label: string;
  q0: number;
  p0: number;
}

export interface ModelMetrics {
  rollout_mse: number;
  median_traj_mse: number;
  mean_abs_energy_drift: number;
  n_params: number;
  train_seconds: number;
  [key: string]: unknown;
}

export interface Manifest {
  trajectories: TrajectoryMeta[];
  models: string[];
  labels: Record<string, string>;
  physics: Record<string, unknown>;
  metrics: Record<string, ModelMetrics>;
}

export const SERIES_ORDER: SeriesKey[] = ["truth", "mlp", "hnn", "node"];

export const COLORS: Record<SeriesKey, string> = {
  truth: "#e6e6e6",
  mlp: "#4aa3df",
  hnn: "#e5534b",
  node: "#3fb950",
};

export const LABELS: Record<SeriesKey, string> = {
  truth: "Ground truth",
  mlp: "MLP",
  hnn: "HNN",
  node: "Neural ODE",
};
