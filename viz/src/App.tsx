import { useEffect } from "react";
import Scene from "./components/Scene";
import Charts from "./components/Charts";
import Controls from "./components/Controls";
import Metrics from "./components/Metrics";
import { useStore } from "./store";

export default function App() {
  const init = useStore((s) => s.init);
  const error = useStore((s) => s.error);
  const loading = useStore((s) => s.loading);

  useEffect(() => {
    void init();
  }, [init]);

  return (
    <div className="app">
      <header>
        <div>
          <h1>Physics-Informed World Models</h1>
          <p>
            HNN vs. Neural ODE vs. MLP on a bouncing ball, where contact breaks
            the smooth-Hamiltonian assumption
          </p>
        </div>
        <a
          className="ghost"
          href="https://arxiv.org/abs/1906.01563"
          target="_blank"
          rel="noreferrer"
        >
          HNN paper
        </a>
      </header>

      {error && (
        <div className="error">
          Could not load rollout data: {error}
          <br />
          <small>Run `python evaluate.py`, then copy viz_data/ into viz/public/data/.</small>
        </div>
      )}

      <main>
        <section className="stage">
          <Scene />
          {loading && !error && <div className="loading">loading rollouts…</div>}
        </section>

        <aside className="panel">
          <Charts />
          <Metrics />
        </aside>
      </main>

      <Controls />

      <footer>
        Rollouts precomputed in PyTorch; the ball you see is each model free-running
        from the same initial state. Ground truth uses velocity-Verlet with
        event-resolved restitution.
      </footer>
    </div>
  );
}
