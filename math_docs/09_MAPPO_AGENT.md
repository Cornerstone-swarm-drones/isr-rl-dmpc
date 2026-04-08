# MARL with MAPPO — Multi-Agent Proximal Policy Optimisation

**Source files:**
- `src/isr_rl_dmpc/agents/mappo_agent.py` — `MAPPOAgent`
- `src/isr_rl_dmpc/gym_env/marl_env.py` — `MARLDMPCEnv`

---

## Table of Contents

1. [MARL Framework Overview](#1-marl-framework-overview)
1. [Centralised Training / Decentralised Execution (CTDE)](#2-centralised-training--decentralised-execution-ctde)
1. [Observation Space (40-D per Agent)](#3-observation-space-40-d-per-agent)
1. [Action Space (14-D per Agent)](#4-action-space-14-d-per-agent)
1. [Reward Function](#5-reward-function)
1. [PPO Policy Gradient](#6-ppo-policy-gradient)
1. [MAPPO — Shared Centralised Critic](#7-mappo--shared-centralised-critic)
1. [Advantage Estimation (GAE)](#8-advantage-estimation-gae)
1. [PPO Clipping Objective](#9-ppo-clipping-objective)
1. [Entropy Regularisation](#10-entropy-regularisation)
1. [Training Loop](#11-training-loop)
1. [Hyperparameters](#12-hyperparameters)
1. [References](#13-references)

---

## 1. MARL Framework Overview

The system uses **Multi-Agent Reinforcement Learning (MARL)** to adaptively
tune the cost parameters of the DMPC layer.  Rather than hand-tuning $Q$ and $R$
matrices, a policy learns from interaction with the environment which cost weights
lead to better mission performance.

**Key design choice:** The RL policy does *not* replace the DMPC; it only adjusts
its cost weights.  This preserves all hard safety guarantees (collision constraints,
LQR stability) provided by the DMPC while adding adaptive behaviour.

```
MARL Policy (MAPPO)
  │
  │  action: (q_scale, r_scale)   ← 14-D per drone
  ▼
DMPC (CVXPY/OSQP)                ← solves QP with Q_eff, R_eff
  │
  │  control: u = [ax, ay, az]   ← 3-D per drone
  ▼
Drone Physics / ADMM Consensus
```

---

## 2. Centralised Training / Decentralised Execution (CTDE)

During **training**, a centralised critic $V_\phi$ observes the **joint state**
of all $N$ drones (full observability), reducing gradient variance:

$$
V_\phi(\boldsymbol{s}) = V_\phi\bigl(\boldsymbol{o}_1, \ldots, \boldsymbol{o}_N\bigr)
$$

During **execution**, each drone's actor $\pi_\theta^{(i)}$ conditions only on its
own **local observation** $\boldsymbol{o}^{(i)}$ (partial observability):

$$
\boldsymbol{a}^{(i)} \sim \pi_\theta\bigl(\cdot \mid \boldsymbol{o}^{(i)}\bigr)
$$

A **shared** actor network (same $\theta$ for all drones) is used, which
exploits permutation symmetry and reduces the number of parameters.

---

## 3. Observation Space (40-D per Agent)

Each drone receives a 40-dimensional local observation vector:

$$
\boldsymbol{o}^{(i)} \in \mathbb{R}^{40}
$$

| Indices | Component | Dim | Description |
| :--- | :--- | :--- | :--- |
| 0–10 | Own DMPC state | 11 | $[\boldsymbol{p}, \boldsymbol{v}, \boldsymbol{a}, \psi, \dot\psi]$ |
| 11–13 | Reference position | 3 | Current waypoint $\boldsymbol{p}^{\text{ref}}$ |
| 14–16 | Reference velocity | 3 | $\boldsymbol{v}^{\text{ref}}$ |
| 17–19 | Tracking error | 3 | $\boldsymbol{e}_p = \boldsymbol{p} - \boldsymbol{p}^{\text{ref}}$ |
| 20–25 | Nearest neighbour relative state | 6 | $[\Delta\boldsymbol{p}, \Delta\boldsymbol{v}]$ to closest drone |
| 26–28 | Mean swarm position offset | 3 | $\bar{\boldsymbol{p}}_{\mathcal{N}} - \boldsymbol{p}^{(i)}$ |
| 29 | Battery level | 1 | Normalised $[0, 1]$ |
| 30 | Health | 1 | Structural health $[0, 1]$ |
| 31–33 | Last applied control | 3 | Previous $\boldsymbol{u}^{(i)}$ |
| 34–36 | ADMM residual | 3 | Primal residual $\lVert\boldsymbol{z}_i - \boldsymbol{v}\rVert$ per axis |
| 37 | DMPC solve time | 1 | Normalised last QP solve time |
| 38 | Collision margin | 1 | Min neighbour distance minus $r_{\min}$ (normalised) |
| 39 | Mission progress | 1 | $t / T_{\max}$ |

---

## 4. Action Space (14-D per Agent)

The action is a 14-dimensional vector of multiplicative cost scale factors:

$$
\boldsymbol{a}^{(i)} = \bigl[\underbrace{q_{s,0}, \ldots, q_{s,10}}_{\boldsymbol{q}_s \in \mathbb{R}^{11}},
  \underbrace{r_{s,0}, r_{s,1}, r_{s,2}}_{\boldsymbol{r}_s \in \mathbb{R}^{3}}\bigr]
\in [0.1,10.0]^{14}
$$

These are passed to the DMPC as:

$$
Q_{\text{eff}}^{(i)} = Q \odot \text{diag}(\boldsymbol{q}_s^{(i)}), \qquad
R_{\text{eff}}^{(i)} = R \odot \text{diag}(\boldsymbol{r}_s^{(i)})
$$

The actor network outputs a mean $\boldsymbol{\mu}_a$ and log-std
$\log\boldsymbol{\sigma}_a$; actions are sampled from

$$
\boldsymbol{a}^{(i)} = \text{clip} \bigl(\boldsymbol{\mu}_a + \boldsymbol{\sigma}_a \odot \boldsymbol{\varepsilon},0.1,10.0\bigr)
  \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, I)
$$

---

## 5. Reward Function

The per-step scalar reward for drone $i$ is a weighted combination:

$$
r^{(i)}_t = w_{\text{track}}r_{\text{track}} + w_{\text{form}}r_{\text{form}} + w_{\text{safe}}r_{\text{safe}} + w_{\text{eff}}r_{\text{eff}}
$$

### Tracking Reward

$$
r_{\text{track}} = \exp \bigl(-\alpha\lVert\boldsymbol{e}_p\rVert^2\bigr) - 1, \quad \alpha = 0.1
$$

Exponential shaping gives dense gradients near the reference.

### Formation Reward

$$
r_{\text{form}} = -\frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)}
  \bigl\lVert\boldsymbol{p}^{(i)} - \boldsymbol{p}^{(j)} - \boldsymbol{d}_{ij}\bigr\rVert
$$

where $\boldsymbol{d}_{ij}$ is the desired relative position in the target formation.

### Safety Penalty

$$
r_{\text{safe}} = \sum_{j \in \mathcal{N}(i)}
  \min \bigl(0, \lVert\boldsymbol{p}^{(i)} - \boldsymbol{p}^{(j)}\rVert - r_{\min}\bigr)
$$

### Efficiency Reward

$$
r_{\text{eff}} = -\lVert\boldsymbol{u}^{(i)}\rVert^2
$$

Penalises unnecessarily large control commands.

### Default Weights

| Weight | Value |
| :--- | :--- |
| $w_{\text{track}}$ | 5.0 |
| $w_{\text{form}}$ | 2.0 |
| $w_{\text{safe}}$ | 10.0 |
| $w_{\text{eff}}$ | 0.1 |

---

## 6. PPO Policy Gradient

Standard policy gradient maximises the expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T} r_t\right]
$$

The gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[
  \sum_{t} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t) \hat{A}_t
\right]
$$

where $\hat{A}_t$ is the **advantage estimate** at step $t$.

---

## 7. MAPPO — Shared Centralised Critic

MAPPO modifies standard PPO by using a **centralised value function** that
conditions on the concatenated observations of all agents:

$$
\hat{A}_t^{(i)} = r_t^{(i)} + \gamma V_\phi\bigl(\boldsymbol{o}_{t+1}^{(1:N)}\bigr) - V_\phi\bigl(\boldsymbol{o}_t^{(1:N)}\bigr)
$$

This global critic significantly reduces advantage variance compared to a
purely local critic, especially when rewards are spatially correlated (as in
formation keeping).

**Network architecture:**

| Network | Input | Hidden layers | Output |
| :--- | :--- | :--- | :--- |
| Actor $\pi_\theta$ | 40-D local obs | 256 → 256 | 28-D ($\boldsymbol{\mu}_a, \log\boldsymbol{\sigma}_a$) |
| Critic $V_\phi$ | $N \times 40$-D joint obs | 256 → 256 | 1-D value |

---

## 8. Advantage Estimation (GAE)

Generalised Advantage Estimation (GAE, Schulman et al. 2016) smoothly
interpolates between TD(1) and Monte-Carlo returns:

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
$$

where the **TD residual** is:

$$
\delta_t = r_t + \gamma V_\phi(\boldsymbol{o}_{t+1}) - V_\phi(\boldsymbol{o}_t)
$$

| Parameter | Value | Effect |
| :--- | :--- | :--- |
| $\gamma$ | 0.99 | Discount factor |
| $\lambda$ | 0.95 | GAE smoothing (bias–variance trade-off) |

---

## 9. PPO Clipping Objective

PPO prevents destructively large policy updates by clipping the probability ratio:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[
  \min \bigl(\rho_t(\theta)\hat{A}_t,
    \text{clip}(\rho_t(\theta),1{-}\epsilon,1{+}\epsilon)\hat{A}_t\bigr)
\right]
$$

where the **importance sampling ratio** is:

$$
\rho_t(\theta) = \frac{\pi_\theta(\boldsymbol{a}_t \mid \boldsymbol{o}_t)}
  {\pi_{\theta_{\text{old}}}(\boldsymbol{a}_t \mid \boldsymbol{o}_t)}
$$

The clip parameter $\epsilon = 0.2$ is the default.

---

## 10. Entropy Regularisation

An entropy bonus prevents premature collapse to a deterministic policy:

$$
\mathcal{L}^{\text{ENT}}(\theta) = \mathbb{E}_t \left[
  \mathcal{H}\bigl[\pi_\theta(\cdot \mid \boldsymbol{o}_t)\bigr]
\right]
$$

For a Gaussian policy $\pi \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2 I)$:

$$
\mathcal{H} = \frac{1}{2}\ln \bigl((2\pi e)^{d} \det(\boldsymbol{\Sigma})\bigr)
  = \sum_{j=1}^{d} \bigl(\log\sigma_j + \tfrac{1}{2}\log(2\pi e)\bigr)
$$

The full PPO-MAPPO objective is:

$$
\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\phi) + c_2 \mathcal{L}^{\text{ENT}}(\theta)
$$

where $\mathcal{L}^{\text{VF}} = \mathbb{E}\_t[(V_\phi(\boldsymbol{o}\_t) - V\_t^{\text{target}})^2]$
is the value function loss, $c_1 = 0.5$, $c_2 = 0.01$.

---

## 11. Training Loop

```
for episode in range(num_episodes):
    obs = env.reset()          # MARLDMPCEnv: N × 40-D
    rollout_buffer.clear()

    for step in range(n_steps):
        actions, values, log_probs = policy.forward(obs)
        # actions: N × 14-D (q_scale, r_scale per drone)

        next_obs, rewards, dones, infos = env.step(actions)
        # env internally runs DMPC + ADMM at 50 Hz for dt seconds

        rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
        obs = next_obs

    # Compute GAE advantages
    rollout_buffer.compute_returns_and_advantage(last_values, dones)

    # PPO update (multiple epochs over the rollout buffer)
    for epoch in range(n_epochs):
        for batch in rollout_buffer.get(batch_size):
            loss = ppo_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
```

---

## 12. Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_steps` | 2048 | Steps per rollout buffer |
| `batch_size` | 256 | Mini-batch size for PPO update |
| `n_epochs` | 10 | PPO update epochs per rollout |
| `lr` | 3 × 10⁻⁴ | Adam learning rate |
| `clip_range` $\epsilon$ | 0.2 | PPO clipping parameter |
| `gamma` $\gamma$ | 0.99 | Discount factor |
| `gae_lambda` $\lambda$ | 0.95 | GAE smoothing parameter |
| `ent_coef` $c_2$ | 0.01 | Entropy coefficient |
| `vf_coef` $c_1$ | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `action_clip` | [0.1, 10.0] | Scale factor bounds |

---

## 13. References

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal
   Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.
1. C. Yu, A. Velu, E. Vinitsky, Y. Wang, A. Bayen, and Y. Wu, "The Surprising
   Effectiveness of PPO in Cooperative Multi-Agent Games,"
   *NeurIPS*, 2022.
1. J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-Dimensional
   Continuous Control Using Generalised Advantage Estimation,"
   *ICLR*, 2016.
1. R. Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
   Environments," *NeurIPS*, 2017.
