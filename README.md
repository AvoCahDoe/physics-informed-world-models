# Physics-Informed World Models: HNN vs. baselines on a bouncing ball

Comparison of a Hamiltonian Neural Network against an MLP and a Neural ODE on a
1D bouncing ball, where floor contact breaks the smooth-Hamiltonian assumption.

State is `z = [q, p]` with `H(q, p) = p^2 / (2m) + m g q`. Between bounces the
motion is Hamiltonian; at the floor the momentum flips with a coefficient of
restitution, `p <- -e p`. With `e = 1` energy is conserved and the HNN's
inductive bias is exactly right; with `e < 1` it is not, and that gap is the
point of the project.

## Setup

Install a CUDA-matched torch first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Training auto-selects CUDA when available; pass `--device cpu` to override.

## Running

```bash
python -m sim.ball                                  # generate datasets into data/
python verify.py                                    # 8 sanity checks, all must pass
python train.py --model mlp  --rollout-k 8 --epochs 1500
python train.py --model hnn  --rollout-k 8 --epochs 1500
python train.py --model node --rollout-k 8 --epochs 1500
python evaluate.py                                  # figures into plots/
```

`evaluate.py` also writes `plots/metrics.json` and `viz_data/` rollouts.
Figure 11 additionally needs the elastic-trained variants:

```bash
python train.py --model hnn --data train_elastic.npz --epochs 1000 \
    --rollout-k 8 --tag hnn_elastic     # likewise for mlp and node
```

## Results

See [results.md](results.md) for every figure with its interpretation. The short
version: continuous-time structure (HNN and Neural ODE) beats the discrete MLP
by ~300x on rollout error, but the Hamiltonian prior does **not** beat a plain
Neural ODE here — once contact is handled explicitly, the remaining free-flight
dynamics are linear and leave the prior nothing to exploit. The HNN wins only in
the elastic case, where energy conservation is exactly true. Separately, no
model survives contact without an explicit event rule.

## Layout

| Path | Purpose |
| --- | --- |
| `sim/ball.py` | Simulator and dataset generation |
| `models/` | `mlp.py`, `hnn.py`, `node.py`, `contact.py` |
| `train.py` | Shared training loop with k-step rollout loss |
| `evaluate.py` | Figures, metrics table, viz export |
| `verify.py` | Sanity checks for the simulator and eval harness |
| `results.md` | Written interpretation of every figure |
| `viz/` | React visualization (deployable static site) |

## Interactive site

`viz/` is a React app with three routes:

| Route | What it is |
| --- | --- |
| `/docs` | The method: system, models, contact handling, training, limitations |
| `/results` | The rollout player plus all 13 figures with interpretations |
| `/try` | A live simulation you can drive |

`/results` animates all three models free-running against ground truth, with
energy and height charts synced to the same clock and a toggle for the naive
(no contact event) rollouts. Those rollouts are precomputed, so the site is
fully static — no backend.

`/try` is not a replay. `viz/src/sim/ball.ts` is a TypeScript port of
`sim/ball.py`, so the browser runs the same velocity-Verlet integrator with the
same exact-crossing contact resolution, and you can vary drop height, initial
velocity, restitution, gravity and mass live.

```bash
cd viz && npm install && npm run dev
```

`evaluate.py` writes `viz_data/` and mirrors both the rollouts and the figures
into `viz/public/`, so the deployed site can't disagree with the checkpoints
that produced them.

### Checks

```bash
npm install            # repo root: puppeteer for the browser tests
npm run verify-sim     # TS simulator vs. Python ground truth
npm run smoke          # headless pass over all three routes
npm run make-og        # regenerate the social share card
```

`verify-sim` re-runs each exported trajectory through the TypeScript port and
requires the Python truth to fall inside the envelope implied by the JSON's
5-decimal rounding, which is a tighter statement than any fixed tolerance.
`smoke` needs `npm run build && npm run preview` running in `viz/`; point it
elsewhere with `SMOKE_URL` to check a deployment. It covers all three routes
and fails on console errors, broken figures, missing share assets, or any
non-200 response.

`make-og` draws `viz/public/og.png` using the TypeScript simulator, so the
trajectory on the share card is real output rather than an illustration.

## Deploying

Live at **https://physics-informed-world-models.vercel.app**.

The app is a static SPA. Both platforms treat `viz/` as the project root, so
there is no repo-root build indirection.

**Vercel** — [viz/vercel.json](viz/vercel.json) is the project config. Deploy
from `viz/`, not the repo root:

```bash
cd viz && npx vercel --prod
```

**Render** — [render.yaml](render.yaml) declares a static site with
`rootDir: viz`. Push the repo, then in Render choose **New > Blueprint** and
point it at the repo; it picks up `render.yaml` automatically.

Both configs fingerprint-cache `/assets/*` forever and cache `/data/*` and
`/plots/*` for an hour, since those are regenerated rather than fingerprinted.

Client-side routing means `/docs`, `/results` and `/try` are not real files, so
both configs rewrite unmatched paths to `index.html`. Static assets are matched
first, so `/data/*` and `/plots/*` still resolve normally.

Two things are worth knowing if you re-create the Vercel project. Don't set
`cleanUrls`, which rewrites `/index.html` and breaks the SPA fallback so every
deep link 404s. And link the project from inside `viz/`: linking from the repo
root makes the CLI create a "services" project, whose routing ignores the
rewrites entirely.

Note that `viz/public/data/` and `viz/public/plots/` must be committed — they
hold the rollouts and figures the deployed site serves. The `.gitignore`
anchors `/data/` to the repo root specifically so these are not ignored.

## Notes

`viz_data/config.json` carries the physics constants, which is what lets
`viz/src/sim/ball.ts` reproduce ground truth exactly rather than approximating
it.
