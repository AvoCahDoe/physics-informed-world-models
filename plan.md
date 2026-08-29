Architecture decision: two-track system

Full MuJoCo contact dynamics + HNN is a known hard combination (energy isn't smooth at contact events, which breaks the pure Hamiltonian assumption). To get a working, honest project fast:

Track A (build first, ~2-3 days): A custom lightweight simulator in Python for a bouncing ball / pendulum-with-wall-collision. Contact is treated as an instantaneous event with a coefficient of restitution. This is fast to simulate, fast to train on, and is exactly the toy problem used in real HNN/LNN papers to demonstrate contact limitations.
Track B (stretch goal, only if time remains): Swap in MuJoCo for a more "real" system (ball dropped on a plane, cartpole with wall). Same model code, different data source.

Starting with Track A means you have full plots by day 2, and you can honestly discuss in your proposal why contact breaks strict energy conservation — which is a sharper, more research-literate insight than a naive "HNN wins" result.

Phase 0 — Environment setup
bash
python -m venv venv && source venv/bin/activate
pip install torch torchdiffeq numpy scipy matplotlib scikit-learn umap-learn
pip install mujoco dm_control  # only needed for Track B

Repo structure to tell Cursor to scaffold:

physical-ai-project/
  sim/           # data generation
  models/        # mlp.py, hnn.py, node.py
  train.py
  evaluate.py
  plots/         # generated figures
  viz/           # React app for interactive sim
  data/
Phase 1 — Data generation (sim/)

Build a bouncing-ball or pendulum-in-a-box simulator with:

State [q, p] (position, momentum) or [theta, theta_dot] for pendulum
Semi-implicit / symplectic Euler integration between contacts
Event detection for wall/floor contact, velocity reversed by coefficient of restitution e < 1 (or e=1 for elastic, so you can ablate energy loss)
Randomize initial conditions, and for the generalization test, randomize mass/restitution per trajectory
Save trajectories as .npz: arrays of (t, q, p) plus the sampled mass/restitution per rollout

Cursor prompt: "Write a Python bouncing ball simulator with symplectic Euler integration, floor collision with coefficient of restitution, that generates N trajectories of length T with randomized initial height/velocity and randomized mass, saved to .npz files."

Phase 2 — Models (models/)
MLP baseline — takes (q, p) (or a short history window), predicts (q', p') directly, trained with MSE.
HNN — network outputs a scalar H(q,p); dynamics come from dq/dt = ∂H/∂p, dp/dt = -∂H/∂q via autograd, integrated with an ODE solver. Reference implementation: greydanus/hamiltonian-nn (github.com/greydanus/hamiltonian-nn) — clone it, adapt the architecture rather than rewriting from scratch.
Neural ODE baseline — a plain dz/dt = f_θ(z) MLP integrated with torchdiffeq.odeint. Reference: rtqichen/torchdiffeq (github.com/rtqichen/torchdiffeq).
Optional 4th model for extra credibility: Lagrangian NN — MilesCranmer/lagrangian_nns (github.com/MilesCranmer/lagrangian_nns), useful if you want to also predict from (q, q_dot) coordinates.

Since contact breaks pure energy conservation, add a contact-aware variant: HNN integrates smoothly between events, and you inject the known restitution rule at collision times. This is the detail that will make a reviewer think "this person understands the failure mode," not just "ran a tutorial."

Phase 3 — Training (train.py)
Same train/val split of trajectories for all three models
Loss: one-step prediction MSE, but also train with multi-step rollout loss (backprop through k integration steps) since that's what actually differentiates the models in long-horizon plots
Log: per-epoch loss, and periodically compute rollout error + energy drift on a held-out trajectory

Cursor prompt: "Write a training loop that trains MLP, HNN, and Neural ODE models on the same trajectory dataset, with an option for k-step rollout loss, logging train/val loss per epoch."

Phase 4 — Evaluation & the 5 plots (evaluate.py)

Use matplotlib for the paper-style versions (these go in your PDF/proposal):

Rollout error vs. horizon (log-y): roll out each model from the same initial state, plot ||pred - true|| vs. timestep for all three, averaged over multiple test trajectories with shaded std.
Energy over time: compute H(q,p) (or 0.5*m*v² + m*g*h) along each model's rollout vs. ground truth — this is your headline plot.
Phase portraits: q vs p, ground truth trajectory overlaid with each model's rollout, for a few representative initial conditions.
Generalization bar chart: train on mass=1kg, test error on mass ∈ {0.5, 0.8, 1.5, 2.0} for each model.
Latent/embedding plot: for the HNN, sample many (q,p) points, get scalar H output, and also grab an intermediate hidden layer, project with UMAP (umap-learn) or PCA (sklearn), color by H value — shows whether the learned energy manifold is smooth and structured.
Phase 5 — React visualization (viz/)

This is what makes it a portfolio piece rather than just static plots — an interactive page where someone can pick initial conditions and watch ground truth vs. each model's rollout animate side by side.

Stack:

react-three-fiber + @react-three/drei — for a real-time 3D (or 2D-in-3D) animated ball/pendulum, since it's the standard for physics-y visualizations in React (docs: docs.pmnd.rs/react-three-fiber)
recharts or Plotly.js via react-plotly.js — for the live-updating energy/phase-portrait plots synced to the animation timestep (recharts: recharts.org; react-plotly: github.com/plotly/react-plotly.js)
zustand — small state store to hold current timestep and drive both the 3D view and the charts off one shared clock (github.com/pmndrs/zustand)

Data flow: run your trained models in Python, export rollouts as JSON arrays of {t, q, p, energy} per model, load them statically into the React app (no need for a live Python backend — keeps deployment trivial, e.g. on Vercel).

Cursor prompt for this phase: "Build a React app with react-three-fiber that animates a ball bouncing based on a precomputed trajectory JSON, with a slider/play button controlling time, and a recharts panel showing energy-over-time for ground truth vs. 3 model predictions, synced to the same time cursor. Support switching between multiple loaded trajectory JSON files."

Phase 6 — Writeup

2-page PDF (LaTeX or Word): problem statement, method, the 5 plots, and a paragraph explicitly connecting it to physics-informed world models for embodied AI, citing 2-3 papers (HNN, LNN, and one world-model paper like Ha & Schmidhuber's "World Models" or a Dreamer paper) so it reads as literature-aware, not a random tutorial project.

All the links

Papers:

Greydanus, Dzamba, Yosinski — Hamiltonian Neural Networks: arxiv.org/abs/1906.01563
Cranmer et al. — Lagrangian Neural Networks: arxiv.org/abs/2003.04630
Chen et al. — Neural Ordinary Differential Equations: arxiv.org/abs/1806.07366
Ha & Schmidhuber — World Models: arxiv.org/abs/1803.10122

Code:

github.com/greydanus/hamiltonian-nn
github.com/MilesCranmer/lagrangian_nns
github.com/rtqichen/torchdiffeq
github.com/google-deepmind/mujoco (Track B)
github.com/google-deepmind/dm_control (Track B)

Frontend:

docs.pmnd.rs/react-three-fiber (3D animation)
recharts.org (charts)
github.com/plotly/react-plotly.js (alternative charting)
github.com/pmndrs/zustand (state)

Python tooling:

umap-learn: umap-learn.readthedocs.io
scikit-learn PCA: scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html