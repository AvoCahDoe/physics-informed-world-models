import { useEffect } from "react";
import Scene from "../components/Scene";
import Charts from "../components/Charts";
import Controls from "../components/Controls";
import Metrics from "../components/Metrics";
import { FIGURES } from "../content/figures";
import { useStore } from "../store";

export default function Results() {
  const init = useStore((s) => s.init);
  const error = useStore((s) => s.error);
  const loading = useStore((s) => s.loading);

  useEffect(() => {
    void init();
  }, [init]);

  return (
    <div className="page">
      <section className="page-head">
        <h1>Results</h1>
        <p className="lede">
          Replay each model free-running from the same initial state, then work
          through the full figure set. Every number comes from 50 held-out test
          trajectories.
        </p>
      </section>

      {error && (
        <div className="error">
          Could not load rollout data: {error}
          <br />
          <small>Run `python evaluate.py` to regenerate viz/public/data/.</small>
        </div>
      )}

      <section className="player">
        <div className="player-grid">
          <div className="stage">
            <Scene />
            {loading && !error && <div className="loading">loading rollouts…</div>}
          </div>
          <aside className="panel">
            <Charts />
            <Metrics />
          </aside>
        </div>
        <Controls />
      </section>

      <section className="page-head">
        <h2>Figure set</h2>
        <p className="lede">
          Thirteen figures, each with what it shows and what it means.
        </p>
      </section>

      <div className="figures">
        {FIGURES.map((f, i) => (
          <figure key={f.id} id={f.id} className="figure">
            <div className="figure-head">
              <span className="figure-num">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <h3>{f.title}</h3>
                <p className="caption">{f.caption}</p>
              </div>
            </div>
            <a href={`/plots/${f.file}`} target="_blank" rel="noreferrer">
              <img src={`/plots/${f.file}`} alt={f.title} loading="lazy" />
            </a>
            <figcaption>
              {f.body.map((p, k) => (
                <p key={k}>{p}</p>
              ))}
              {f.takeaway && <p className="takeaway">{f.takeaway}</p>}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
