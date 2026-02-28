import React, { useState } from "react";

/* ── Section data ─────────────────────────────────────────────── */
const SECTIONS = [
  {
    title: "Drone Kinematics",
    icon: "🚁",
    overview: `Each drone moves in 3D space. At every time step the RL policy outputs velocity commands
(vx, vy, vz) that are clipped to the drone's physical limits. The position updates as:

  position(t+1) = position(t) + velocity(t) × Δt

where Δt = 1 / control_frequency. Yaw angle θ is updated similarly using a yaw-rate command ω:

  θ(t+1) = θ(t) + ω(t) × Δt

Velocity and acceleration are bounded:
  |v| ≤ v_max ,   |a| ≤ a_max`,
    detail: `**Full State Vector**
Each drone i has state  sᵢ = [x, y, z, vx, vy, vz, θ, ω, battery]ᵀ ∈ ℝ⁹.

The discrete-time dynamics follow a double-integrator model with Euler integration:

  pᵢ(t+1) = pᵢ(t) + vᵢ(t)·Δt + ½·aᵢ(t)·Δt²
  vᵢ(t+1) = vᵢ(t) + aᵢ(t)·Δt

subject to the constraints:
  ‖vᵢ‖₂ ≤ v_max          (velocity limit)
  ‖aᵢ‖₂ ≤ a_max          (acceleration limit)
  |ωᵢ| ≤ ω_max            (yaw rate limit)

**Battery Model**
Energy consumed per step is proportional to thrust:
  Eᵢ(t) = c₁·‖aᵢ‖² + c₂·‖vᵢ‖ + c₃    (hovering baseline c₃)

Battery state:
  bᵢ(t+1) = bᵢ(t) − Eᵢ(t)·Δt

When bᵢ ≤ 0 the drone is considered dead and returns to base.

**Rotation Dynamics (Simplified)**
For attitude control the system uses a quaternion representation internally:
  q(t+1) = q(t) ⊗ Δq(ω, Δt)

where Δq is the incremental rotation quaternion from the angular velocity vector.`,
  },
  {
    title: "Swarm Formation Control",
    icon: "🔗",
    overview: `The drones maintain a desired formation using a graph-based approach. Each drone is a node and
communication links are edges. The formation is preserved when the algebraic connectivity λ₂
(second-smallest eigenvalue of the graph Laplacian) stays positive.

  If λ₂ > 0 → the swarm is connected.
  If λ₂ = 0 → the swarm is fragmented.

A formation error penalty pushes drones toward desired relative positions.`,
    detail: `**Graph Laplacian**
Let G = (V, E) be the communication graph where V = {1, …, N} drones and edge (i,j) ∈ E iff
‖pᵢ − pⱼ‖ ≤ R_comm.

The adjacency matrix A has entries  aᵢⱼ = 1  if (i,j) ∈ E, else 0.
The degree matrix D = diag(d₁, …, dₙ) where dᵢ = Σⱼ aᵢⱼ.
The Laplacian is  L = D − A.

**Algebraic Connectivity**
Eigenvalues of L: 0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ.
λ₂ (Fiedler value) measures how well the graph is connected.

**Formation Error**
Given desired relative positions Δpᵢⱼ* between drone i and j, the formation error is:

  e_form = (1/|E|) Σ_{(i,j)∈E} ‖(pᵢ − pⱼ) − Δpᵢⱼ*‖²

**Collision Avoidance Constraint**
  ‖pᵢ − pⱼ‖ ≥ d_safe   ∀ i ≠ j

This is enforced as a hard constraint in DMPC and as a penalty in the RL reward.`,
  },
  {
    title: "Distributed Model Predictive Control (DMPC)",
    icon: "🎛️",
    overview: `DMPC generates safe, constraint-respecting trajectories for each drone. Each drone solves a local
optimization problem over a prediction horizon T steps:

  minimize   Σ cost(state, control)   over T future steps
  subject to  dynamics, collision avoidance, communication maintenance

Drones exchange planned trajectories and re-optimize in a receding-horizon fashion. This ensures
safety constraints (min separation, max speed) are always met.`,
    detail: `**DMPC Formulation for Drone i**

  min_{uᵢ(0), …, uᵢ(T−1)}   Σₜ₌₀ᵀ⁻¹ [ ‖xᵢ(t) − xᵢ*(t)‖²_Q + ‖uᵢ(t)‖²_R ] + ‖xᵢ(T) − xᵢ*(T)‖²_P

subject to:
  xᵢ(t+1) = Axᵢ(t) + Buᵢ(t)                  (linear dynamics)
  ‖uᵢ(t)‖∞ ≤ u_max                              (input bounds)
  ‖xᵢ(t) − xⱼ(t)‖ ≥ d_safe + εₜ     ∀j ≠ i   (tightened collision avoidance)
  ‖xᵢ(t) − xⱼ(t)‖ ≤ R_comm           ∀j ∈ Nᵢ  (communication maintenance)

where Q, R, P are weight matrices, xᵢ* is the reference from the RL policy, and εₜ is the
constraint tightening parameter that increases over the horizon for robustness.

**Receding Horizon**
At each real time step:
1. Solve the optimization for T steps ahead.
2. Apply only the first control input uᵢ(0).
3. Shift the horizon forward by one step.
4. Re-solve with updated neighbor trajectories.

**Solver**
The problem is formulated as a Quadratic Program (QP) and solved using CVXPY with the OSQP backend.
Max iterations and tolerance are configurable (default: 200 iters, 1e-4 tol).`,
  },
  {
    title: "Reinforcement Learning (SAC)",
    icon: "🧠",
    overview: `The high-level drone policy is trained using Soft Actor-Critic (SAC), an off-policy RL algorithm
that maximizes both the expected reward and the entropy of the policy:

  Objective = E[ Σ γᵗ (reward(t) + α · entropy) ]

The policy network (actor) outputs drone velocity commands. A value network (critic) estimates
how good each state is. The algorithm balances exploration (trying new actions) with exploitation
(using what works).`,
    detail: `**SAC Objective**
  J(π) = Σₜ E_{(sₜ,aₜ)~ρ_π} [ r(sₜ, aₜ) + α H(π(·|sₜ)) ]

where:
  - π is the policy,  r is the reward,  γ is the discount factor
  - H(π) = −E[log π(a|s)] is the entropy
  - α is the temperature parameter (auto-tuned)

**Actor (Policy Network)**
  π_φ(a|s) = tanh(μ_φ(s) + σ_φ(s) · ε),    ε ~ N(0, I)

The actor outputs mean μ and std σ for a Gaussian, squashed through tanh to bound actions.

**Twin Critics (Q-Networks)**
Two Q-networks Q_{θ₁}, Q_{θ₂} are trained to minimize:
  L(θᵢ) = E[ (Q_{θᵢ}(s,a) − y)² ]

where the target is:
  y = r + γ (min(Q_{θ'₁}(s',a'), Q_{θ'₂}(s',a')) − α log π(a'|s'))

**Observation Space** (per drone):
  - Own position, velocity, heading [7 dims]
  - Relative positions of neighbors [N × 3 dims]
  - Sensor readings (radar, optical, RF detections) [variable]
  - Coverage map (local grid) [G × G dims]
  - Battery level [1 dim]

**Action Space** (per drone):
  - Velocity command (vx, vy, vz) ∈ [−v_max, v_max]³
  - Yaw rate ω ∈ [−ω_max, ω_max]

**Training Hyperparameters**
  - γ = 0.99 (discount factor)
  - τ = 0.005 (soft target update rate)
  - Replay buffer size: 1,000,000
  - Batch size: 256
  - Actor/Critic LR: 3e-4 / 3e-4`,
  },
  {
    title: "Reward Function",
    icon: "🏆",
    overview: `The reward at each time step combines multiple objectives:

  R(t) = w₁·R_coverage + w₂·R_energy + w₃·R_collision + w₄·R_target + w₅·R_formation + w₆·R_comm

Each component encourages a different behavior:
  • Coverage: reward for exploring new grid cells
  • Energy: penalty for excessive power usage
  • Collision: large penalty for getting too close to other drones
  • Target engagement: reward for detecting and classifying targets
  • Formation: penalty for deviating from desired formation
  • Communication: penalty for breaking communication links`,
    detail: `**Detailed Reward Components**

**Coverage Reward**
  R_cov = (cells_covered(t) − cells_covered(t−1)) / total_cells

This rewards incremental coverage improvement.

**Energy Penalty**
  R_energy = −c_e · Σᵢ (‖vᵢ‖² + ‖aᵢ‖²) / N

Normalized by number of drones N.

**Collision Penalty**
  R_col = −c_col · Σᵢ Σⱼ>ᵢ max(0, d_safe − ‖pᵢ − pⱼ‖)²

Quadratic penalty that increases sharply as drones get closer.

**Target Engagement Reward**
  R_target = c_tgt · Σₖ I(target_k detected) · confidence_k

where I(·) is the indicator function.

**Formation Penalty**
  R_form = −c_form · e_form    (see Formation Control section)

**Communication Penalty**
  R_comm = −c_comm · max(0, −λ₂)

Penalizes loss of algebraic connectivity.

**Default Weights** (from learning_config.yaml):
  w_coverage = 1.0,  w_energy = 0.1,  w_collision = 5.0
  w_target = 0.5,  w_formation = 0.3,  w_communication = 0.2`,
  },
  {
    title: "Sensor Models & Target Classification",
    icon: "📡",
    overview: `Each drone carries three sensors:

  • Radar – long range (200 m), detects position but not identity
  • Optical – medium range (120° FOV), provides visual classification
  • RF Sensor – detects radio emissions (100 m range)

Sensor readings are fused using an Extended Kalman Filter (EKF) to estimate target states.
A Random Forest classifier labels targets as hostile, friendly, or unknown based on fused features.`,
    detail: `**Radar Model**
Detection probability follows a range-dependent model:
  P_detect(r) = P₀ · exp(−(r/R_max)²)

Measurement noise is Gaussian:  z_radar = p_target + N(0, σ_r²·I)

**Optical Model**
Detection requires the target to be within the camera's FOV cone:
  cos(angle_to_target) ≥ cos(FOV/2)

Classification features are extracted from the simulated image patch.

**Extended Kalman Filter (EKF)**
State: x = [px, py, pz, vx, vy, vz]ᵀ

Prediction:
  x̂(t|t−1) = F·x̂(t−1|t−1)
  P(t|t−1)  = F·P(t−1|t−1)·Fᵀ + Q

Update (for each sensor measurement z):
  K = P(t|t−1)·Hᵀ·(H·P(t|t−1)·Hᵀ + R)⁻¹
  x̂(t|t) = x̂(t|t−1) + K·(z − H·x̂(t|t−1))
  P(t|t) = (I − K·H)·P(t|t−1)

**Random Forest Classifier**
Features: [radar_rcs, speed, altitude, rf_signature, optical_size, optical_shape_ratio]
Classes: {hostile, friendly, unknown, neutral}
Trained on labeled examples; confidence = average tree vote fraction.`,
  },
  {
    title: "Coverage & Area Discretization",
    icon: "🗺️",
    overview: `The mission area is divided into a grid of square cells (default 10 m × 10 m). A cell is marked as
"covered" when a drone's sensor footprint overlaps it.

  Coverage(t) = (# covered cells) / (total cells)

The mission succeeds when Coverage(t) ≥ coverage_goal (default 95%). A revisit timer tracks how
long since each cell was last observed.`,
    detail: `**Grid Representation**
For an area of size W × H with cell size c:
  Grid dimensions:  Gx = ⌈W/c⌉,  Gy = ⌈H/c⌉
  Total cells: Gx × Gy

**Sensor Footprint**
Drone i at position pᵢ with altitude hᵢ covers a circular area:
  Coverage radius: r_cov(hᵢ) = hᵢ · tan(FOV/2)

All cells within distance r_cov of pᵢ's ground projection are marked as covered.

**Revisit Tracking**
  τ_cell(t) = t − t_last_visited(cell)

If τ_cell > revisit_interval, the cell is marked as "stale" and needs re-observation.

**Coverage Efficiency Metric**
  η = Coverage(T_final) / (N_drones × T_final × v_max × r_cov)

Measures how efficiently the drones cover the area relative to their theoretical maximum.`,
  },
];

/* ── Component ────────────────────────────────────────────────── */
export default function MathView() {
  const [showDetail, setShowDetail] = useState(SECTIONS.map(() => false));

  const toggleDetail = (idx) => {
    setShowDetail((prev) => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  };

  return (
    <div>
      <div className="card" style={{ marginBottom: 20 }}>
        <h3>📐 Math &amp; Physics Reference</h3>
        <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
          This section explains the mathematics and physics behind the ISR-RL-DMPC system.
          Each topic starts with a <strong style={{ color: "var(--accent-green)" }}>simplified overview</strong> suitable
          for presentations, with an option to expand into <strong style={{ color: "var(--accent-purple)" }}>full mathematical detail</strong>.
        </p>
      </div>

      {SECTIONS.map((sec, idx) => (
        <div className="card math-section" key={idx}>
          <h3>{sec.icon} {sec.title}</h3>

          {/* Overview – always visible */}
          <div className="math-overview">
            <div className="math-label" style={{ color: "var(--accent-green)" }}>Overview</div>
            <pre className="math-block">{sec.overview}</pre>
          </div>

          {/* Toggle detail */}
          <button
            className="btn"
            style={{ marginTop: 8, fontSize: 10, padding: "4px 14px" }}
            onClick={() => toggleDetail(idx)}
          >
            {showDetail[idx] ? "▾ Hide Detailed Math" : "▸ Show Detailed Math"}
          </button>

          {showDetail[idx] && (
            <div className="math-detail">
              <div className="math-label" style={{ color: "var(--accent-purple)" }}>Detailed Formulation</div>
              {sec.detail.split("\n").map((line, i) => {
                const trimmed = line;
                if (!trimmed.trim()) return <br key={i} />;
                /* Bold markers */
                const parts = trimmed.split(/\*\*(.*?)\*\*/g);
                const isBold = /^\*\*/.test(trimmed.trim());
                return (
                  <p
                    key={i}
                    className={isBold ? "math-heading" : "math-line"}
                    style={{ margin: "2px 0" }}
                  >
                    {parts.map((part, j) =>
                      j % 2 === 1
                        ? <strong key={j} style={{ color: "var(--accent-cyan)" }}>{part}</strong>
                        : <span key={j}>{part}</span>
                    )}
                  </p>
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
