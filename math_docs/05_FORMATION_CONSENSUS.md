# Formation Control and Distributed Consensus

**Source file:**
- `src/isr_rl_dmpc/modules/formation_controller.py` — `ConsensusController`, `FormationGeometry`, `FormationController`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Graph Theory Foundations](#2-graph-theory-foundations)
3. [Consensus Protocol](#3-consensus-protocol)
4. [Formation Control Law](#4-formation-control-law)
5. [Formation Geometries](#5-formation-geometries)
   - 5.1 [Line](#51-line)
   - 5.2 [Wedge (V-Shape)](#52-wedge-v-shape)
   - 5.3 [Column](#53-column)
   - 5.4 [Circular](#54-circular)
   - 5.5 [Grid](#55-grid)
   - 5.6 [Sphere (Fibonacci)](#56-sphere-fibonacci)
6. [Convergence Analysis](#6-convergence-analysis)
7. [Formation Quality Metrics](#7-formation-quality-metrics)

---

## 1  Overview

The Formation Controller (Module 2) coordinates the swarm into prescribed
geometric patterns using a **distributed consensus protocol**.  Each drone
adjusts its velocity based only on local information (its own state plus
the states of its communication neighbours), without a central coordinator.

The controller produces a **control acceleration** for each drone:

```
u_i = u_p + u_d + u_consensus
```

where:
- `u_p` — proportional position error term
- `u_d` — velocity damping term
- `u_consensus` — consensus correction from neighbours

---

## 2  Graph Theory Foundations

### Communication Graph

The swarm is modelled as an undirected graph `G = (V, E)`:
- **V** = {0, 1, …, N−1} — drone indices
- **(i, j) ∈ E** iff `‖p_i − p_j‖ ≤ r_comm` (communication range, default 100 m)

### Adjacency Matrix

The weighted adjacency matrix **A_adj** has entries:

```
A_adj[i, j] = 1   if (i,j) ∈ E and i ≠ j
A_adj[i, j] = 0   otherwise
```

In the code this is computed by `StateManager.get_inter_drone_adjacency(comm_range)`.

### Graph Laplacian

The **graph Laplacian** L = D − A_adj, where **D** is the degree matrix:

```
D[i, i] = Σ_j A_adj[i, j]   (degree of node i)
L[i, j] = −A_adj[i, j]      for i ≠ j
```

**Key property:** For a **connected graph**, L has exactly one zero eigenvalue
(corresponding to the consensus direction) and all other eigenvalues are
strictly positive:

```
0 = λ₀ < λ₁ ≤ λ₂ ≤ … ≤ λ_{N-1}
```

The second-smallest eigenvalue λ₁ (the **algebraic connectivity** or
**Fiedler value**) determines the speed of consensus convergence.

---

## 3  Consensus Protocol

### Position Consensus Error

Let `d_{ij}` be the desired relative position between drones i and j in
formation.  The **formation error** for drone i is:

```
ε_i = Σ_{j ∈ 𝒩(i)} (p_i − p_j − d_{ij})
```

When `ε_i → 0` for all i, the swarm is in the desired formation.

### Standard Consensus Update

The consensus term in the control law drives formation error to zero:

```
u_consensus_i = k_c Σ_{j ∈ 𝒩(i)} (p_j − p_i)
              = −k_c (L p)_i
```

where `k_c = 0.1` is the consensus gain and `(L p)_i` is the i-th row of the
Laplacian matrix applied to the position vector.

This attracts drone i towards its neighbours' centroid, reducing positional
spread while preserving average position.

---

## 4  Formation Control Law

For drone **i** with desired position `p_des_i`, current position `p_i`,
and velocity `v_i`:

### Proportional Term

```
e_p = p_des_i − p_i     (position error)

u_p = k_p · e_p,   k_p = 2.0
```

### Derivative (Damping) Term

```
u_d = −k_d · v_i,   k_d = velocity_damping (default 0.1)
```

This term suppresses oscillations and reduces overshoot.

### Consensus Term

```
p_avg = (1 / |𝒩(i)|) Σ_{j ∈ 𝒩(i)} p_j    (neighbour centroid)

u_consensus = k_c · (p_avg − p_i),   k_c = 0.1
```

### Combined Control

```
u_total = u_p + u_d + u_consensus

‖u_total‖ is saturated at u_max = 5.0 m/s²
```

---

## 5  Formation Geometries

Each formation type generates the desired position `p_des_i` for drone `i`
as an offset from the formation **center** `c ∈ ℝ³` rotated by **heading** `θ`.

### 5.1  Line

Drones are evenly spaced along the heading direction:

```
offset_i = (i − (N−1)/2) × spacing

p_des_i = c + [offset_i · cos(θ),   offset_i · sin(θ),   z_i]
```

where `z_i = (i − (N−1)/2) × z_spacing` stacks layers in altitude.

| Parameter | Default |
|-----------|---------|
| `spacing` | 10 m |
| `z_spacing` | 2 m |

### 5.2  Wedge (V-Shape)

The lead drone (`i = 0`) flies at the center; remaining drones alternate
left and right behind:

```
r_i = spacing × ⌈i/2⌉     (radial distance increases with rank)
φ_i = (i/N) π              (opening angle)
side_i = (−1)^i            (alternates left/right)

p_des_i = c + [ r_i cos(θ + side_i φ_i),
                r_i sin(θ + side_i φ_i),
                (i mod 2) × z_spacing ]
```

The wedge formation optimises sensor coverage and forward-looking threat
detection for ISR missions.

### 5.3  Column

Single-file queue along the heading direction:

```
p_des_i = c + [i · spacing · cos(θ),   i · spacing · sin(θ),   0]
```

### 5.4  Circular

Drones equally spaced on a circle of radius `r = scale / 2`:

```
φ_i = 2π i / N + θ       (azimuth for drone i)

p_des_i = c + [r cos(φ_i),   r sin(φ_i),   0]
```

**Perimeter coverage** — maximises sensor footprint for area surveillance.

### 5.5  Grid

Drones arranged in a `⌈√N⌉ × ⌈√N⌉` lattice, rotated by heading:

```
col_i = i mod grid_size,   row_i = i ÷ grid_size

x = (col_i − grid_size/2) × spacing
y = (row_i − grid_size/2) × spacing

p_des_i = c + [x cos(θ) − y sin(θ),   x sin(θ) + y cos(θ),   0]
```

### 5.6  Sphere (Fibonacci)

3-D spherical distribution using the **Fibonacci sphere** algorithm, which
gives near-uniform coverage of a sphere surface:

```
φ_i = arccos(1 − 2i/N)       (polar angle, i = 0, …, N−1)
θ_i = √(N π) × φ_i           (azimuth, golden angle spiral)
r   = scale / 2

p_des_i = c + [ r sin(φ_i) cos(θ_i),
                r sin(φ_i) sin(θ_i),
                r cos(φ_i) ]
```

The golden-angle spacing ensures the minimum angular distance between any two
drones is approximately `√(4π/N)` steradians, optimising 3-D coverage.

---

## 6  Convergence Analysis

### Linear Consensus Convergence

For the pure consensus update `ẋ = −L x` (continuous time), the convergence
rate is governed by the algebraic connectivity:

```
‖x(t) − x*‖ ≤ e^{−λ₁ t} ‖x(0) − x*‖
```

where `λ₁ > 0` is the Fiedler value of L and `x*` is the consensus value.

### Discrete-Time Convergence

In discrete time with step Δt:

```
x[k+1] = (I − Δt k_c L) x[k]

convergence rate: ρ = 1 − Δt k_c λ₁ < 1
```

The condition `Δt k_c λ_max(L) < 2` must hold for convergence.  With
`k_c = 0.1`, `Δt = 0.02 s`, and typical `λ_max(L) < 10`, the product is
`0.002 × 10 = 0.02 ≪ 2`, so convergence is guaranteed.

### Formation Convergence Check

Convergence threshold `ε = 0.5 m` (default):

```python
for drone_id, desired_pos in desired_positions.items():
    error = ‖drone_state.position − desired_pos‖
    if error > threshold:
        return False
return True  # converged
```

---

## 7  Formation Quality Metrics

```
errors = [‖p_i − p_des_i‖ for i = 0, …, N−1]

mean_error = (1/N) Σ errors_i
max_error  = max(errors)
std_error  = std(errors)
rmse       = sqrt((1/N) Σ errors_i²)
```

These metrics are computed by `ConsensusController.get_formation_metrics()` and
can be logged or used to trigger formation type changes during a mission.
