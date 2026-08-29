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

## Interactive visualization

`viz/` is a React + react-three-fiber app that animates all three models
free-running against ground truth, with energy and height charts synced to the
same clock, and a toggle for the naive (no contact event) rollouts.

```bash
cd viz && npm install && npm run dev
```

The rollouts are precomputed, so the site is fully static — no backend.
`evaluate.py` writes `viz_data/` and mirrors it into `viz/public/data/`.

## Deploying

The app is a static SPA; both platforms build from the repo root using the
committed configs.

**Vercel** — [vercel.json](vercel.json) sets the build to `viz/` and publishes
`viz/dist`:

```bash
npx vercel --prod
```

**Render** — [render.yaml](render.yaml) declares a static site with
`rootDir: viz`. Push the repo, then in Render choose **New > Blueprint** and
point it at the repo; it picks up `render.yaml` automatically.

Both configs fingerprint-cache `/assets/*` forever and cache `/data/*` for an
hour, since rollout JSON is regenerated rather than fingerprinted.

Note that `viz/public/data/` must be committed — it holds the rollouts the
deployed site serves. The `.gitignore` anchors `/data/` to the repo root
specifically so this directory is not ignored.

## Notes

`viz_data/config.json` carries the physics constants so a future React app can
port the simulator to JS and reproduce ground truth exactly.
