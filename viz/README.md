# Interactive visualization

React app that animates ground truth and all three models free-running from the
same initial state, with charts synced to a single shared clock.

- **react-three-fiber** for the 3D balls and floor
- **recharts** for energy and height, sharing a time cursor with the animation
- **zustand** as the one clock driving both

## Data

Rollouts are precomputed in PyTorch and served as static JSON from
`public/data/` — there is no backend. `python evaluate.py` writes them and
mirrors them here automatically.

`public/data/config.json` carries the physics constants (`g`, `dt`, `e`) so the
simulator can be ported to JS later and reproduce ground truth exactly.

## Develop

```bash
npm install
npm run dev
```

## Test

The headless smoke test lives here but its `puppeteer` dependency is installed
at the repo root, keeping deploy installs free of a Chromium download:

```bash
npm run build && npm run preview -- --port 4173   # terminal 1
cd .. && npm install && npm run smoke              # terminal 2
```

It fails on any console error and checks the canvas, charts, metrics, and that
the clock advances. It writes `smoke.png` and `smoke_naive.png`.

## Note on the "Naive" toggle

Switching contact handling to **Naive** replays rollouts with no contact event
injected. Every model falls straight through the floor: a smooth vector field
cannot represent an instantaneous velocity reversal. That is the project's main
finding, not a bug in the app.
