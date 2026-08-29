import { create } from "zustand";
import type { Manifest, SeriesKey, Trajectory } from "./types";
import { SERIES_ORDER } from "./types";

const BASE = import.meta.env.BASE_URL;

interface State {
  manifest: Manifest | null;
  trajectory: Trajectory | null;
  selectedId: number;
  /** Single shared clock: the 3D view and every chart read this one value. */
  frame: number;
  playing: boolean;
  speed: number;
  mode: "contact" | "naive";
  visible: Record<SeriesKey, boolean>;
  loading: boolean;
  error: string | null;

  init: () => Promise<void>;
  select: (id: number) => Promise<void>;
  setFrame: (f: number) => void;
  advance: (dtSeconds: number) => void;
  togglePlay: () => void;
  setSpeed: (s: number) => void;
  setMode: (m: "contact" | "naive") => void;
  toggleSeries: (k: SeriesKey) => void;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}data/${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const useStore = create<State>((set, get) => ({
  manifest: null,
  trajectory: null,
  selectedId: 0,
  frame: 0,
  playing: true,
  speed: 1,
  mode: "contact",
  visible: { truth: true, mlp: true, hnn: true, node: true },
  loading: true,
  error: null,

  init: async () => {
    try {
      const manifest = await getJSON<Manifest>("index.json");
      set({ manifest });
      await get().select(manifest.trajectories[0]?.id ?? 0);
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  select: async (id) => {
    set({ loading: true });
    try {
      const trajectory = await getJSON<Trajectory>(`trajectory_${id}.json`);
      set({ trajectory, selectedId: id, frame: 0, loading: false, error: null });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  setFrame: (f) => {
    const n = get().trajectory?.t.length ?? 1;
    set({ frame: Math.max(0, Math.min(n - 1, Math.round(f))) });
  },

  advance: (dtSeconds) => {
    const { trajectory, speed, frame } = get();
    if (!trajectory) return;
    // Advance in simulation time so playback rate is independent of frame rate.
    const step = (dtSeconds * speed) / trajectory.dt;
    const next = frame + step;
    set({ frame: next >= trajectory.t.length - 1 ? 0 : next });
  },

  togglePlay: () => set((s) => ({ playing: !s.playing })),
  setSpeed: (speed) => set({ speed }),
  setMode: (mode) => set({ mode }),
  toggleSeries: (k) =>
    set((s) => ({ visible: { ...s.visible, [k]: !s.visible[k] } })),
}));

/** Series actually drawn, respecting the visibility toggles. */
export function activeSeries(visible: Record<SeriesKey, boolean>): SeriesKey[] {
  return SERIES_ORDER.filter((k) => visible[k]);
}
