import { Link } from "react-router-dom";

export default function Docs() {
  return (
    <article className="page prose">
      <section className="hero">
        <h1>Does a Hamiltonian prior help when physics has contact?</h1>
        <p className="lede">
          Three world models — an MLP, a Hamiltonian Neural Network, and a Neural
          ODE — learn the dynamics of a bouncing ball from the same data, the
          same training loop, and the same evaluation harness. The bounce is what
          makes it interesting: a velocity discontinuity is not something a
          smooth vector field can represent.
        </p>
        <div className="hero-actions">
          <Link className="btn primary" to="/results">
            See the results
          </Link>
          <Link className="btn" to="/try">
            Run a simulation
          </Link>
        </div>
      </section>

      <section>
        <h2>The headline</h2>
        <p>
          Both continuous-time models beat the discrete MLP by roughly{" "}
          <strong>300x</strong> on rollout MSE. But the HNN does{" "}
          <strong>not</strong> beat the plain Neural ODE — they are within noise
          of each other, and the Neural ODE is slightly ahead while training 15%
          faster. The interesting result is <em>why</em>, and it follows directly
          from the contact structure.
        </p>
        <div className="callout">
          What mattered was <strong>continuous-time</strong> structure, not{" "}
          <strong>Hamiltonian</strong> structure. Once contact is handled
          explicitly, the remaining problem is free flight under constant
          gravity, whose vector field is linear in the state — close to the
          easiest possible target for any approximator. There is almost no room
          for a structural prior to add value.
        </div>
      </section>

      <section>
        <h2>The system</h2>
        <p>
          A ball in 1D under gravity with a floor at <code>q = 0</code>. State is{" "}
          <code>z = (q, p)</code>: height and momentum. Between contacts the
          dynamics are Hamiltonian with
        </p>
        <pre className="eq">H(q, p) = p² / 2m + mgq</pre>
        <p>
          At a contact the momentum flips and loses a fraction of its magnitude,{" "}
          <code>p ← −e·p</code>, where <code>e</code> is the coefficient of
          restitution. This rule is instantaneous, so it sits outside the smooth
          flow entirely. With <code>e = 1</code> energy is exactly conserved and
          the system is genuinely Hamiltonian; with <code>e &lt; 1</code> it is
          not.
        </p>

        <h3>Ground truth integrator</h3>
        <p>
          Free flight uses <strong>velocity Verlet</strong>, which is exact for a
          constant force, so energy is conserved to machine precision (relative
          spread 1.1e-13 over 4000 steps). Contact is resolved at the exact
          floor-crossing time — the positive root of the flight parabola — rather
          than at the end of the timestep. An earlier semi-implicit Euler version
          drifted 13.5% in the elastic case, which would have contaminated every
          energy metric with an integrator artefact.
        </p>
      </section>

      <section>
        <h2>The three models</h2>
        <p>
          All three have ~41k parameters and are deliberately matched so the
          comparison isolates the structural assumption.
        </p>

        <div className="cards">
          <div className="card">
            <h3 style={{ color: "#1f77b4" }}>MLP</h3>
            <p className="tag">Discrete map</p>
            <p>
              Learns the next state directly: <code>z ← z + f(z)</code> over one
              training timestep. No notion of an underlying continuous flow, so
              nothing constrains its errors to be consistent between steps.
            </p>
          </div>
          <div className="card">
            <h3 style={{ color: "#d62728" }}>HNN</h3>
            <p className="tag">Hamiltonian vector field</p>
            <p>
              Outputs a <em>scalar</em> <code>H(q, p)</code>, and derives the
              dynamics from its gradient via{" "}
              <code>dq/dt = ∂H/∂p, dp/dt = −∂H/∂q</code>. Energy conservation is
              structural, not learned. The second-order autograd this needs costs
              10x the MLP's training time.
            </p>
          </div>
          <div className="card">
            <h3 style={{ color: "#2ca02c" }}>Neural ODE</h3>
            <p className="tag">Unconstrained vector field</p>
            <p>
              Learns <code>dz/dt = f(z)</code> directly and integrates it. Shares
              the HNN's continuous-time structure but none of its Hamiltonian
              constraint — which makes it the control that isolates what the
              prior is actually contributing.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2>Contact-aware rollout</h2>
        <p>
          Every model is rolled out twice. The <strong>naive</strong> rollout
          just integrates the learned field. The{" "}
          <strong>contact-aware</strong> rollout detects a floor crossing within
          a step, integrates exactly up to the crossing time, applies{" "}
          <code>p ← −e·p</code>, and integrates the remainder.
        </p>
        <p>
          The naive version fails catastrophically for all three models — the
          ball falls through the floor and keeps going. That is not an
          optimisation failure but a representational one, and it is the sharpest
          finding in the project. You can toggle it yourself on the{" "}
          <Link to="/results">results page</Link>.
        </p>
      </section>

      <section>
        <h2>Training</h2>
        <ul>
          <li>
            200 training trajectories at mass 1.0 kg, <code>dt = 0.01 s</code>, 2
            s horizon, restitution <code>e = 0.8</code>; 50 held-out test
            trajectories.
          </li>
          <li>
            Loss combines a one-step derivative term with a{" "}
            <strong>k-step contact-aware rollout</strong> term, so the models are
            optimised for the thing they are evaluated on.
          </li>
          <li>
            Derivative targets come from central differences with
            contact-adjacent samples <strong>masked out</strong> — the derivative
            is undefined across a discontinuity, and including those samples
            teaches the model to smear the bounce.
          </li>
          <li>
            Splits are by <strong>trajectory</strong>, not by timestep, so
            validation is a true holdout rather than interpolation between
            neighbouring frames.
          </li>
        </ul>
      </section>

      <section>
        <h2>Honest limitations</h2>
        <ol>
          <li>
            The HNN's advantage over the Neural ODE is not demonstrated here; it
            is confined to the elastic energy metric.
          </li>
          <li>
            No model generalises across mass, because mass is not an input. That
            figure measures a design choice, not a property of the priors.
          </li>
          <li>
            The restitution coefficient is <em>given</em> to the rollout, not
            learned. A stronger version would infer it from data.
          </li>
          <li>
            Long-horizon rollouts converge to a spurious non-zero state instead
            of coming to rest.
          </li>
          <li>
            The system is 1D with an analytically simple flight phase. A pendulum
            or a ball with drag would give the Hamiltonian prior something to
            actually do.
          </li>
        </ol>
      </section>

      <section>
        <h2>Reproducing</h2>
        <pre className="code">
          <code>{`pip install -r requirements.txt

python -m sim.ball     # generate datasets
python verify.py       # sanity checks, all must pass
python train.py --model mlp  --rollout-k 8 --epochs 1500
python train.py --model hnn  --rollout-k 8 --epochs 1500
python train.py --model node --rollout-k 8 --epochs 1500
python evaluate.py     # every figure + metrics.json + viz data`}</code>
        </pre>
        <p>
          Training auto-detects CUDA and falls back to CPU with a printed reason,
          so a silent 50x slowdown can't go unnoticed.{" "}
          <code>evaluate.py</code> mirrors its figures and rollouts into this
          app's assets, so the deployed site can never show results that
          disagree with the checkpoints that produced them.
        </p>
      </section>
    </article>
  );
}
