# Coverage Path Planning — Grid Decomposition and Waypoint Generation

**Source file:**
- `src/isr_rl_dmpc/modules/mission_planner.py` — `GridDecomposer`, `WaypointGenerator`, `MissionPlanner`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
1. [Grid Decomposition](#2-grid-decomposition)
1. [Cell Priority Scoring](#3-cell-priority-scoring)
1. [Cell-to-Drone Assignment](#4-cell-to-drone-assignment)
1. [Waypoint Generation](#5-waypoint-generation)
   - 5.1 [Nearest-Neighbour Path](#51-nearest-neighbour-path)
   - 5.2 [Sweep (Boustrophedon) Path](#52-sweep-boustrophedon-path)
   - 5.3 [Spiral Path](#53-spiral-path)
1. [Mission Time Estimation](#6-mission-time-estimation)
1. [Coverage Tracking](#7-coverage-tracking)
1. [Complexity Analysis](#8-complexity-analysis)

---

## 1. Problem Statement

Given:
- A mission area defined by a polygon $\mathcal{B} \subset \mathbb{R}^2$ (a set of boundary vertices)
- A swarm of $n$ drones with positions $\boldsymbol{p}_1, \ldots, \boldsymbol{p}_n \in \mathbb{R}^2$
- A grid resolution $r$ (metres per cell side)
- A sensor coverage radius $\rho_s$ (metres)

**Goal:** Generate a waypoint sequence for each drone such that:
1. Every point in $\mathcal{B}$ is visited by at least one drone (≥ 95 % coverage).
1. The total path length and mission duration are minimised.
1. Higher-priority areas are covered first.

This is a **Coverage Path Planning (CPP)** problem, which is NP-hard in
general.  The system uses a greedy grid-based decomposition combined with
nearest-neighbour tour planning for real-time deployment.

---

## 2. Grid Decomposition

### Uniform Grid

The mission area is over-laid with an axis-aligned uniform grid of square
cells of side $r$:

$$
x_{\text{grid}} = [x_{\min}, x_{\min} + r, x_{\min} + 2r, \ldots, x_{\max}]
$$

$$
y_{\text{grid}} = [y_{\min}, y_{\min} + r, y_{\min} + 2r, \ldots, y_{\max}]
$$

A cell with vertices $\{(x_i, y_j), (x_{i+1}, y_j), (x_{i+1}, y_{j+1}), (x_i, y_{j+1})\}$
is included in the mission if its **centroid**
$\boldsymbol{c} = \left(\frac{x_i + x_{i+1}}{2}, \frac{y_j + y_{j+1}}{2}\right)$
lies inside the mission boundary polygon $\mathcal{B}$.

### Point-in-Polygon Test

Membership $\boldsymbol{c} \in \mathcal{B}$ is decided by `GeometryOps.point_in_polygon(c, B)`, which
uses the **ray-casting algorithm**:

> Cast a horizontal ray from $\boldsymbol{c}$ to $(+\infty, c_y)$.
> Count the number of boundary edges it crosses.
> $\boldsymbol{c} \in \mathcal{B} \iff$ crossing count is odd.

This runs in $O(|\mathcal{B}|)$ per point and is robust to non-convex polygons.

### Cell Area

For a square grid cell with resolution $r$:

$$
A_{\text{cell}} = r^2 \quad (\text{m}^2)
$$

---

## 3. Cell Priority Scoring

Each cell is assigned a priority $\pi_c \in [0.1, 1.0]$ that determines the order
in which cells are visited.  Higher priority → visited earlier in the mission.

### Gaussian Distance-from-Centre Priority

The priority decreases exponentially with distance from the area centre:

$$
\boldsymbol{c}_{\text{area}} = \text{mean}(\mathcal{B}), \qquad
d_c = \lVert\boldsymbol{c} - \boldsymbol{c}_{\text{area}}\rVert, \qquad
s = \lVert\max(\mathcal{B}) - \min(\mathcal{B})\rVert
$$

$$
\pi_c = \exp\!\left(-\frac{d_c}{s / 10}\right), \qquad
\pi_c \leftarrow \text{clip}(\pi_c, 0.1, 1.0)
$$

**Rationale:** ISR missions typically originate from a headquarters position
near the area centre; central regions are higher-value targets.  The $s/10$
factor controls the priority falloff radius.

---

## 4. Cell-to-Drone Assignment

Cells are assigned to drones using **greedy nearest-neighbour assignment**,
applied in priority-sorted order:

1. Sort cells by priority (descending).
1. For each cell $c$ (in priority order):

$$
j^* = \arg\min_{j=1,\ldots,n} \lVert\boldsymbol{c}_{\text{center}} - \boldsymbol{p}_j[:2]\rVert, \qquad
\text{Assign } c \to \text{drone } j^*
$$

This is a **greedy spatial partition**: each cell goes to the nearest drone,
so drones naturally cover geographically close regions.  It does not guarantee
equal workload distribution, but it produces short individual paths.

**Complexity:** $O(m \log m)$ for sorting $+ O(mn)$ for assignment,
where $m$ = number of cells and $n$ = number of drones.

---

## 5. Waypoint Generation

After assignment, each drone $d$ receives a list of cells $\{c_1, c_2, \ldots, c_k\}$.
The waypoint generator orders these cells and returns a 3-D waypoint array
$W \in \mathbb{R}^{k \times 3}$:

$$
W[i] = \bigl[c_i.\text{center}[0], c_i.\text{center}[1], \text{altitude}\bigr]
$$

Three path strategies are implemented:

### 5.1. Nearest-Neighbour Path

A **greedy tour** starting from the drone's current position:

> $\text{remaining} = \{c_1, \ldots, c_k\}$, $\text{path} = []$, $\text{current} = \boldsymbol{p}_{\text{start}}$
>
> While $\text{remaining} \ne \emptyset$:
> $c^* = \arg\min_{c \in \text{remaining}} \lVert c.\text{center} - \text{current}\rVert$
> append $c^{\*}$ to path, set $\text{current} = c^\*.\text{center}$, remove $c^\*$ from remaining.

This is the classic **nearest-neighbour heuristic** for the Travelling Salesman
Problem (TSP).  It is not optimal in general, but runs in $O(k^2)$ and typically
gives tours within 20–25% of the optimum.

### 5.2. Sweep (Boustrophedon) Path

Cells are sorted by $y$-coordinate first, then $x$ within each $y$-band:

$$
\text{path} = \text{sorted}\bigl(\text{cells}, \text{key} = \lambda c:(c.\text{center}[1], c.\text{center}[0])\bigr)
$$

The resulting path resembles a **lawnmower** (boustrophedon) pattern: the drone
sweeps left-to-right across one row, then right-to-left across the next.  This
minimises backtracking for rectangular mission areas.

### 5.3. Spiral Path

Cells are sorted by distance from the centroid of all assigned cells,
from nearest to farthest:

$$
\boldsymbol{\mu} = \text{mean}\bigl(\{c.\text{center}\}\bigr), \qquad
\text{path} = \text{sorted}\bigl(\text{cells}, \text{key} = \lambda c:\lVert c.\text{center} - \boldsymbol{\mu}\rVert\bigr)
$$

This visits the highest-priority (central) cells first and spirals outward —
useful when early coverage of the centre is critical.

---

## 6. Mission Time Estimation

Given a waypoint sequence $W = [\boldsymbol{w}\_0, \boldsymbol{w}\_1, \ldots, \boldsymbol{w}\_{k-1}]$ with hover time $h$
per waypoint and drone cruise speed $v$:

$$
L = \sum_{i=0}^{k-2} \lVert\boldsymbol{w}_{i+1} - \boldsymbol{w}_i\rVert
  \quad \text{(total path length)}
$$

$$
T_{\text{travel}} = L / v, \qquad
T_{\text{hover}} = k \cdot h
$$

$$
T_{\text{total}} = (T_{\text{travel}} + T_{\text{hover}}) \times 1.2
  \quad \text{(20\% buffer)}
$$

The 20% buffer accounts for turns, altitude changes, and acceleration phases
not captured by the constant-speed model.

**Edge case:** If $k = 1$, there is no travel (only hover):

$$
T_{\text{total}} = T_{\text{hover}} \times 1.2
$$

---

## 7. Coverage Tracking

A **boolean coverage matrix** tracks which cells have been visited:

$$
M_{\text{cov}} \in \{\text{false}, \text{true}\}^m
$$

A cell is marked as covered when the drone passes within $\rho_s$ metres of its
centroid (sensor footprint check).  The coverage ratio is:

$$
\eta = \frac{\sum_c M_{\text{cov}}[c]}{m} \in [0, 1]
$$

The mission continues until $\eta \ge \eta_{\text{goal}}$ (default 0.95)
or the maximum mission duration is reached.

---

## 8. Complexity Analysis

| Operation | Complexity |
| :--- | :--- |
| Grid decomposition | $O(m_{\text{total}} \cdot \lvert\mathcal{B}\rvert)$ where $m_{\text{total}} = (x_{\text{range}}/r) \times (y_{\text{range}}/r)$ |
| Point-in-polygon per cell | $O(\lvert\mathcal{B}\rvert)$ |
| Priority scoring | $O(m)$ |
| Cell-to-drone assignment | $O(m \log m + mn)$ |
| Nearest-neighbour tour | $O(k^2)$ per drone |
| Sweep / spiral tour | $O(k \log k)$ per drone |

For a 500×500 m area with $r = 20\text{m}$ and $n = 4$ drones:
- $m \approx 625$ cells, $k \approx 156$ cells per drone
- Nearest-neighbour tour: ~24,000 distance computations per drone — negligible
  on modern hardware (< 1 ms).
