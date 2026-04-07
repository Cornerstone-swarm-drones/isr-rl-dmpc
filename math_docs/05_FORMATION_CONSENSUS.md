# Formation Control and Distributed Consensus

**Source files:**
- `src/isr_rl_dmpc/modules/formation_controller.py` — `ConsensusController`, `FormationGeometry`, `FormationController`
- `src/isr_rl_dmpc/modules/admm_consensus.py` — `ADMMConsensus`

---

## Table of Contents

1. [Overview](#1-overview)
1. [Graph Theory Foundations](#2-graph-theory-foundations)
1. [Consensus Protocol](#3-consensus-protocol)
1. [Formation Control Law](#4-formation-control-law)
1. [Formation Geometries](#5-formation-geometries)
   - 5.1 [Line](#51-line)
   - 5.2 [Wedge (V-Shape)](#52-wedge-v-shape)
   - 5.3 [Column](#53-column)
   - 5.4 [Circular](#54-circular)
   - 5.5 [Grid](#55-grid)
   - 5.6 [Sphere (Fibonacci)](#56-sphere-fibonacci)
1. [ADMM Consensus Layer](#6-admm-consensus-layer)
1. [Convergence Analysis](#7-convergence-analysis)
1. [Formation Quality Metrics](#8-formation-quality-metrics)

---

## 1. Overview

The Formation Controller (Module 2) coordinates the swarm into prescribed
geometric patterns using a **distributed consensus protocol**.  Each drone
adjusts its velocity based only on local information (its own state plus
the states of its communication neighbours), without a central coordinator.

The controller produces a **control acceleration** for each drone:

$$
\mathbf{u}_i = \mathbf{u}_p + \mathbf{u}_d + \mathbf{u}_{\text{consensus}}
$$

where:
- $\mathbf{u}_p$ — proportional position error term
- $\mathbf{u}_d$ — velocity damping term
- $\mathbf{u}_{\text{consensus}}$ — consensus correction from neighbours

The ADMM layer (see [10_ADMM_CONSENSUS.md](10_ADMM_CONSENSUS.md)) further
tightens inter-drone agreement by coupling each drone's DMPC sub-problem
through a shared consensus variable.

---

## 2. Graph Theory Foundations

### Communication Graph

The swarm is modelled as an undirected graph $G = (V, E)$:
- $V = \{0, 1, \ldots, N{-}1\}$ — drone indices
- $(i, j) \in E$ iff $\|\mathbf{p}_i - \mathbf{p}_j\| \le r_{\text{comm}}$ (communication range, default 100 m)

### Adjacency Matrix

The weighted adjacency matrix $A_{\text{adj}}$ has entries:

$$
A_{\text{adj}}[i, j] = \begin{cases} 1 & (i,j) \in E,\; i \ne j \\ 0 & \text{otherwise} \end{cases}
$$

### Graph Laplacian

The **graph Laplacian** $L = D - A_{\text{adj}}$, where $D$ is the degree matrix:

$$
D[i, i] = \sum_j A_{\text{adj}}[i, j], \qquad
L[i, j] = -A_{\text{adj}}[i, j] \quad (i \ne j)
$$

**Key property:** For a **connected graph**, $L$ has exactly one zero eigenvalue
and all other eigenvalues are strictly positive:

$$
0 = \lambda_0 < \lambda_1 \le \lambda_2 \le \cdots \le \lambda_{N-1}
$$

The second-smallest eigenvalue $\lambda_1$ (the **algebraic connectivity** or
**Fiedler value**) determines the speed of consensus convergence.

---

## 3. Consensus Protocol

### Position Consensus Error

Let $\mathbf{d}_{ij}$ be the desired relative position between drones $i$ and $j$ in
formation.  The **formation error** for drone $i$ is:

$$
\boldsymbol{\varepsilon}_i = \sum_{j \in \mathcal{N}(i)} (\mathbf{p}_i - \mathbf{p}_j - \mathbf{d}_{ij})
$$

When $\boldsymbol{\varepsilon}_i \to \mathbf{0}$ for all $i$, the swarm is in the desired formation.

### Standard Consensus Update

The consensus term in the control law drives formation error to zero:

$$
\mathbf{u}_{\text{consensus},i} = k_c \sum_{j \in \mathcal{N}(i)} (\mathbf{p}_j - \mathbf{p}_i)
= -k_c\,(L\,\mathbf{p})_i
$$

where $k_c = 0.1$ is the consensus gain and $(L\,\mathbf{p})_i$ is the $i$-th row of the
Laplacian matrix applied to the position vector.

---

## 4. Formation Control Law

For drone $i$ with desired position $\mathbf{p}^{\text{des}}_i$, current position
$\mathbf{p}_i$, and velocity $\mathbf{v}_i$:

### Proportional Term

$$
\mathbf{e}_p = \mathbf{p}^{\text{des}}_i - \mathbf{p}_i, \qquad
\mathbf{u}_p = k_p\,\mathbf{e}_p, \quad k_p = 2.0
$$

### Derivative (Damping) Term

$$
\mathbf{u}_d = -k_d\,\mathbf{v}_i, \quad k_d = 0.1\;\text{(velocity\_damping)}
$$

This term suppresses oscillations and reduces overshoot.

### Consensus Term

$$
\bar{\mathbf{p}}_{\mathcal{N}} = \frac{1}{|\mathcal{N}(i)|}
  \sum_{j \in \mathcal{N}(i)} \mathbf{p}_j, \qquad
\mathbf{u}_{\text{consensus}} = k_c\,(\bar{\mathbf{p}}_{\mathcal{N}} - \mathbf{p}_i), \quad k_c = 0.1
$$

### Combined Control

$$
\mathbf{u}_{\text{total}} = \mathbf{u}_p + \mathbf{u}_d + \mathbf{u}_{\text{consensus}}
$$

$\|\mathbf{u}_{\text{total}}\|$ is saturated at $u_{\max} = 5.0\;\text{m/s}^2$.

---

## 5. Formation Geometries

Each formation type generates the desired position $\mathbf{p}^{\text{des}}_i$ for
drone $i$ as an offset from the formation **center** $\mathbf{c} \in \mathbb{R}^3$
rotated by **heading** $\theta$.

### 5.1. Line

Drones are evenly spaced along the heading direction:

$$
o_i = \left(i - \frac{N{-}1}{2}\right) \times s, \qquad
\mathbf{p}^{\text{des}}_i = \mathbf{c} + \begin{bmatrix} o_i \cos\theta \\ o_i \sin\theta \\ z_i \end{bmatrix}
$$

where $z_i = \left(i - \frac{N{-}1}{2}\right) \times s_z$ stacks layers in altitude,
$s = 10\;\text{m}$, $s_z = 2\;\text{m}$.

### 5.2. Wedge (V-Shape)

The lead drone ($i = 0$) flies at the center; remaining drones alternate
left and right behind:

$$
r_i = s \cdot \lceil i/2 \rceil, \qquad
\varphi_i = \frac{i}{N}\,\pi, \qquad
\text{side}_i = (-1)^i
$$

$$
\mathbf{p}^{\text{des}}_i = \mathbf{c} + \begin{bmatrix}
  r_i \cos(\theta + \text{side}_i\,\varphi_i) \\
  r_i \sin(\theta + \text{side}_i\,\varphi_i) \\
  (i \bmod 2) \times s_z
\end{bmatrix}
$$

### 5.3. Column

Single-file queue along the heading direction:

$$
\mathbf{p}^{\text{des}}_i = \mathbf{c} + \begin{bmatrix} i\,s\cos\theta \\ i\,s\sin\theta \\ 0 \end{bmatrix}
$$

### 5.4. Circular

Drones equally spaced on a circle of radius $r = \text{scale}/2$:

$$
\varphi_i = \frac{2\pi i}{N} + \theta, \qquad
\mathbf{p}^{\text{des}}_i = \mathbf{c} + \begin{bmatrix} r\cos\varphi_i \\ r\sin\varphi_i \\ 0 \end{bmatrix}
$$

### 5.5. Grid

Drones arranged in a $\lceil\sqrt{N}\rceil \times \lceil\sqrt{N}\rceil$ lattice,
rotated by heading:

$$
c_i = i \bmod g, \quad r_i = \lfloor i/g \rfloor, \qquad
x = (c_i - g/2)\,s, \quad y = (r_i - g/2)\,s
$$

$$
\mathbf{p}^{\text{des}}_i = \mathbf{c} + \begin{bmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \\ 0 \end{bmatrix}
$$

### 5.6. Sphere (Fibonacci)

3-D spherical distribution using the **Fibonacci sphere** algorithm:

$$
\varphi_i = \arccos\!\left(1 - \frac{2i}{N}\right), \qquad
\theta_i = \sqrt{N\pi}\,\varphi_i, \qquad
r = \frac{\text{scale}}{2}
$$

$$
\mathbf{p}^{\text{des}}_i = \mathbf{c} + r\begin{bmatrix}
  \sin\varphi_i\cos\theta_i \\
  \sin\varphi_i\sin\theta_i \\
  \cos\varphi_i
\end{bmatrix}
$$

The golden-angle spacing ensures the minimum angular distance between any two
drones is approximately $\sqrt{4\pi/N}$ steradians, optimising 3-D coverage.

---

## 6. ADMM Consensus Layer

The formation consensus protocol is reinforced by the ADMM layer, which
couples each drone's DMPC sub-problem to a shared consensus variable $\mathbf{v}$
(see [10_ADMM_CONSENSUS.md](10_ADMM_CONSENSUS.md)).

The ADMM augmented Lagrangian for the formation consensus problem is:

$$
\mathcal{L}_\rho = \sum_{i=1}^{N} \left[ \|\mathbf{p}_i - \mathbf{p}^{\text{des}}_i\|^2 + \boldsymbol{\mu}_i^\top (\mathbf{p}_i - \mathbf{v}) + \frac{\rho}{2}\|\mathbf{p}_i - \mathbf{v}\|^2 \right]
$$

The global consensus variable $\mathbf{v}$ converges to the swarm centroid:

$$
\mathbf{v}^{k+1} = \frac{1}{N}\sum_{i=1}^{N}
  \bigl(\mathbf{p}_i^{k+1} + \boldsymbol{\mu}_i^k / \rho\bigr)
$$

This ensures that the planned positions from individual DMPC solvers are
globally consistent, preventing the "plan disagreement" that arises when
drones optimise independently.

---

## 7. Convergence Analysis

### Linear Consensus Convergence

For the pure consensus update $\dot{\mathbf{x}} = -L\,\mathbf{x}$ (continuous time), the
convergence rate is governed by the algebraic connectivity:

$$
\|\mathbf{x}(t) - \mathbf{x}^*\| \le e^{-\lambda_1 t}\,\|\mathbf{x}(0) - \mathbf{x}^*\|
$$

where $\lambda_1 > 0$ is the Fiedler value of $L$ and $\mathbf{x}^*$ is the
consensus value.

### Discrete-Time Convergence

In discrete time with step $\Delta t$:

$$
\mathbf{x}[k{+}1] = (I - \Delta t\,k_c\,L)\,\mathbf{x}[k]
$$

$$
\text{convergence rate: } \rho = 1 - \Delta t\,k_c\,\lambda_1 < 1
$$

The condition $\Delta t\,k_c\,\lambda_{\max}(L) < 2$ must hold for convergence.
With $k_c = 0.1$, $\Delta t = 0.02\;\text{s}$, and typical $\lambda_{\max}(L) < 10$,
the product is $0.002 \times 10 = 0.02 \ll 2$, so convergence is guaranteed.

### Formation Convergence Check

Convergence threshold $\varepsilon = 0.5\;\text{m}$ (default):

```python
for drone_id, desired_pos in desired_positions.items():
    error = np.linalg.norm(drone_state.position - desired_pos)
    if error > threshold:
        return False
return True  # converged
```

---

## 8. Formation Quality Metrics

$$
\text{errors}_i = \|\mathbf{p}_i - \mathbf{p}^{\text{des}}_i\|, \quad i = 0, \ldots, N{-}1
$$

$$
\bar{e} = \frac{1}{N}\sum_i \text{errors}_i, \qquad
e_{\max} = \max_i \text{errors}_i, \qquad
\text{RMSE} = \sqrt{\frac{1}{N}\sum_i \text{errors}_i^2}
$$

These metrics are computed by `ConsensusController.get_formation_metrics()` and
can be logged or used to trigger formation type changes during a mission.
