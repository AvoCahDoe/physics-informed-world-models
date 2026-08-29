/** Figure catalogue for /results. Interpretations condensed from results.md,
 *  which stays the canonical long-form writeup. */

export interface Figure {
  id: string;
  file: string;
  title: string;
  caption: string;
  /** Paragraphs of interpretation. */
  body: string[];
  /** Optional pull-quote highlighted alongside the figure. */
  takeaway?: string;
}

export const FIGURES: Figure[] = [
  {
    id: "rollout-error",
    file: "01_rollout_error.png",
    title: "Rollout error vs. horizon",
    caption: "Median trajectory MSE with interquartile band, over a 2 s rollout.",
    body: [
      "The MLP separates from the other two within the first 0.25 s and stays two orders of magnitude worse for the rest of the rollout. All three curves show a staircase: error jumps at bounces and grows slowly in between, because a bounce amplifies whatever positional error already existed — hitting the floor at slightly the wrong time changes the entire subsequent trajectory.",
      "HNN and Neural ODE track each other almost exactly. This is the first sign that the Hamiltonian prior is not buying anything here.",
    ],
  },
  {
    id: "energy",
    file: "02_energy.png",
    title: "Energy over time",
    caption:
      "One representative trajectory (left) and median absolute energy error across the test set (right).",
    body: [
      "Ground truth is a staircase: perfectly flat during free flight, dropping by a factor e² = 0.64 in kinetic energy at each bounce.",
      "The MLP visibly gains energy during free flight — the segments slope upward where they should be flat. It has no structural reason to conserve anything, so its one-step errors accumulate as a systematic energy injection. The HNN and Neural ODE sit on top of ground truth at this scale, with mean absolute energy error smaller by 123x and 227x respectively.",
    ],
    takeaway:
      "The clearest picture of the MLP's failure mode: it manufactures energy during smooth flight.",
  },
  {
    id: "phase",
    file: "03_phase_portraits.png",
    title: "Phase portraits",
    caption: "Trajectories in (q, p) space.",
    body: [
      "Each inward spiral is the bounce sequence losing energy. The vertical segments at q = 0 are the instantaneous momentum flips. The MLP's spiral is visibly displaced from truth after the first bounce; HNN and Neural ODE overlay it.",
    ],
  },
  {
    id: "generalization",
    file: "04_generalization.png",
    title: "Generalization to unseen mass",
    caption: "Rollout MSE at masses the models were never trained on.",
    body: [
      "At the trained mass (1.0 kg) the HNN and Neural ODE are ~300x better than the MLP. At every other mass, all three collapse to roughly the same poor error.",
      "The reason is structural, not a training failure: the models take only (q, p) as input, and mass never appears. The true dynamics dq/dt = p/m genuinely differ at a different mass, so no amount of physical prior can recover it from (q, p) alone. Conditioning the models on mass is the obvious fix and the natural next experiment.",
    ],
    takeaway:
      "The clearest negative result — and it measures a design choice, not a property of the priors.",
  },
  {
    id: "latent",
    file: "05_hnn_latent.png",
    title: "HNN latent structure",
    caption: "Penultimate-layer activations on a (q, p) grid, projected with PCA and UMAP.",
    body: [
      "Both projections are smooth, continuous sheets with a monotone energy gradient and no fragmentation — the network has learned a well-ordered energy manifold rather than memorising samples. The striations are the grid sampling, not structure in the model.",
    ],
  },
  {
    id: "vector-fields",
    file: "06_vector_fields.png",
    title: "Learned vector fields",
    caption: "Learned field over ground truth (top) and log-scale field error (bottom).",
    body: [
      "The dark low-error region is a triangle, and that triangle is exactly the physically reachable set — states satisfying p²/2 + gq ≤ E_max for the initial energies in the training data. Outside it, every model degrades by two to three orders of magnitude.",
      "This is what the aggregate error curves hide: the models are not uniformly accurate over state space, they are accurate exactly where data lives.",
    ],
    takeaway: "Accuracy is confined to the reachable set. Aggregate metrics hide this.",
  },
  {
    id: "hamiltonian",
    file: "07_learned_hamiltonian.png",
    title: "Learned Hamiltonian",
    caption: "The HNN's scalar output against the analytic H = p²/2m + mgq.",
    body: [
      "The HNN recovers the analytic Hamiltonian to a residual standard deviation of 0.0017 J on-distribution, against 2.19 J over the full plotted box. Since H is only defined up to an additive constant, the offset is matched before comparing.",
      "So the HNN genuinely learns the right energy function where it has data. Its failure to outperform the Neural ODE is not a failure to learn physics.",
    ],
  },
  {
    id: "naive-vs-contact",
    file: "08_naive_vs_contact.png",
    title: "Naive vs. contact-aware rollout",
    caption: "Dashed: no contact handling. Solid: explicit restitution at detected crossings.",
    body: [
      "Every model, without exception, fails completely without explicit contact handling. In the naive rollouts the ball passes through the floor and falls forever. A continuous vector field is a smooth map, and no smooth map can produce an instantaneous velocity reversal. This is a representational impossibility, not an optimisation problem, and no amount of training data fixes it.",
      "The solid lines inject the known restitution rule p ← −e·p at detected crossings. This is the single change that makes the task solvable at all.",
    ],
    takeaway:
      "Contact breaks any smooth-flow model, HNN or not — a velocity discontinuity is outside the hypothesis class.",
  },
  {
    id: "drift",
    file: "09_drift_decomposition.png",
    title: "Where the energy error comes from",
    caption: "Accumulated energy error split into the flight phase and contact events, log scale.",
    body: [
      "The MLP acquires 6.85 J in flight and 5.79 J at contact — it is bad everywhere. The HNN takes 0.087 J in flight and 0.047 J at contact; the Neural ODE 0.067 J and 0.046 J.",
      "For the structured models most of the remaining error accrues during smooth flight, not at the bounces. Once restitution is injected exactly, contact stops being the bottleneck — so the way to improve these models further is better flight integration, not better contact handling.",
    ],
  },
  {
    id: "training",
    file: "10_training_curves.png",
    title: "Training curves",
    caption: "Train and validation loss per epoch.",
    body: [
      "All three converge. The MLP's loss is noisier and plateaus two orders higher. The cost asymmetry matters: the HNN takes 817 s against the MLP's 80 s, a 10x penalty for the second-order autograd needed to differentiate its own output — and it does not buy accuracy over the Neural ODE's 703 s.",
    ],
  },
  {
    id: "elastic",
    file: "11_elastic_ablation.png",
    title: "Elastic vs. inelastic ablation",
    caption: "Energy under e = 1.0 (left) and e = 0.8 (right).",
    body: [
      "The cleanest isolation of the Hamiltonian assumption. On the left true energy is exactly conserved and the system is genuinely Hamiltonian; on the right it is not.",
      "The MLP drifts upward by ~14% over 2 s in the elastic case, manufacturing energy from nothing. The HNN and Neural ODE both hold the conserved value flat. Notably the HNN's elastic drift (0.0029 J) is lower than the Neural ODE's (0.0049 J).",
    ],
    takeaway:
      "The one metric where the Hamiltonian prior wins — and it wins precisely where its assumption is exactly true.",
  },
  {
    id: "long-horizon",
    file: "12_long_horizon.png",
    title: "Long-horizon stability",
    caption: "Rolled out to 10x the training horizon (20 s vs 2 s).",
    body: [
      "No model diverges — divergence rate 0.00 for all three, which is a real strength of the contact-aware formulation.",
      "But there is an honest failure here. Ground truth decays to exactly zero as the ball comes to rest, while all three models plateau at a spurious floor and keep jittering forever. None of them learned that rest is an absorbing state — unsurprisingly, since the 2 s training horizon never contained a settled ball.",
    ],
  },
  {
    id: "error-distribution",
    file: "13_error_distribution.png",
    title: "Per-trajectory error distribution",
    caption: "Violin plot of rollout MSE across held-out trajectories.",
    body: [
      "Mean rollout MSE and median differ by three orders of magnitude, so the mean is set by a handful of outlier trajectories, not typical behaviour. The HNN and Neural ODE bodies sit near 1e-6 with a long upper tail. Those outliers are trajectories where a small timing error near a bounce shifted the contact to a different timestep, after which the two trajectories are simply out of phase.",
      "The MLP's distribution is tight and uniformly bad, which is a different kind of failure.",
    ],
  },
];
