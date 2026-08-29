import { useStore } from "../store";
import { COLORS, LABELS, SERIES_ORDER, type SeriesKey } from "../types";

const SPEEDS = [0.25, 0.5, 1, 2];

export default function Controls() {
  const {
    manifest,
    trajectory,
    selectedId,
    frame,
    playing,
    speed,
    mode,
    visible,
    select,
    setFrame,
    togglePlay,
    setSpeed,
    setMode,
    toggleSeries,
  } = useStore();

  const n = trajectory?.t.length ?? 1;
  const t = trajectory?.t[Math.min(Math.floor(frame), n - 1)] ?? 0;

  return (
    <div className="controls">
      <div className="row">
        <button className="play" onClick={togglePlay}>
          {playing ? "Pause" : "Play"}
        </button>
        <input
          className="scrub"
          type="range"
          min={0}
          max={n - 1}
          step={1}
          value={Math.floor(frame)}
          onChange={(e) => setFrame(Number(e.target.value))}
        />
        <span className="time">{t.toFixed(2)} s</span>
      </div>

      <div className="row wrap">
        <div className="group">
          <label>Speed</label>
          <div className="segmented">
            {SPEEDS.map((s) => (
              <button
                key={s}
                className={speed === s ? "on" : ""}
                onClick={() => setSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        <div className="group">
          <label>Contact handling</label>
          <div className="segmented">
            <button
              className={mode === "contact" ? "on" : ""}
              onClick={() => setMode("contact")}
            >
              Contact-aware
            </button>
            <button
              className={mode === "naive" ? "on" : ""}
              onClick={() => setMode("naive")}
            >
              Naive
            </button>
          </div>
        </div>

        <div className="group grow">
          <label>Initial condition</label>
          <select
            value={selectedId}
            onChange={(e) => select(Number(e.target.value))}
          >
            {manifest?.trajectories.map((tr) => (
              <option key={tr.id} value={tr.id}>
                {tr.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="row wrap">
        {SERIES_ORDER.map((k: SeriesKey) => (
          <button
            key={k}
            className={`chip ${visible[k] ? "on" : ""}`}
            onClick={() => toggleSeries(k)}
            style={{ ["--c" as string]: COLORS[k] }}
          >
            <span className="dot" />
            {LABELS[k]}
          </button>
        ))}
      </div>

      {mode === "naive" && (
        <p className="warn">
          Naive rollout: no contact event is injected, so the learned smooth
          vector field cannot reverse the velocity and every model falls through
          the floor. Ground truth is unchanged.
        </p>
      )}
    </div>
  );
}
