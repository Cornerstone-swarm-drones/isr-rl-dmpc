# Coverage Path Planning — Grid Decomposition and Waypoint Generation

**Source file:**
- `src/isr_rl_dmpc/modules/mission_planner.py` — `GridDecomposer`, `WaypointGenerator`, `MissionPlanner`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Grid Decomposition](#2-grid-decomposition)
3. [Cell Priority Scoring](#3-cell-priority-scoring)
4. [Cell-to-Drone Assignment](#4-cell-to-drone-assignment)
5. [Waypoint Generation](#5-waypoint-generation)
   - 5.1 [Nearest-Neighbour Path](#51-nearest-neighbour-path)
   - 5.2 [Sweep (Boustrophedon) Path](#52-sweep-boustrophedon-path)
   - 5.3 [Spiral Path](#53-spiral-path)
6. [Mission Time Estimation](#6-mission-time-estimation)
7. [Coverage Tracking](#7-coverage-tracking)
8. [Complexity Analysis](#8-complexity-analysis)

---

## 1  Problem Statement

Given:
- A mission area defined by a polygon `B ⊂ ℝ²` (a set of boundary vertices)
- A swarm of **n** drones with positions `p_1, …, p_n ∈ ℝ²`
- A grid resolution `r` (metres per cell side)
- A sensor coverage radius `ρ_s` (metres)

**Goal:** Generate a waypoint sequence for each drone such that:
1. Every point in B is visited by at least one drone (≥ 95 % coverage).
2. The total path length and mission duration are minimised.
3. Higher-priority areas are covered first.

This is a **Coverage Path Planning (CPP)** problem, which is NP-hard in
general.  The system uses a greedy grid-based decomposition combined with
nearest-neighbour tour planning for real-time deployment.

---

## 2  Grid Decomposition

### Uniform Grid

The mission area is over-laid with an axis-aligned uniform grid of square
cells of side `r`:

```
x_grid = [x_min, x_min + r, x_min + 2r, …, x_max]
y_grid = [y_min, y_min + r, y_min + 2r, …, y_max]
```

A cell with vertices `{(x_i, y_j), (x_{i+1}, y_j), (x_{i+1}, y_{j+1}), (x_i, y_{j+1})}`
is included in the mission if its **centroid** `c = ((x_i + x_{i+1})/2, (y_j + y_{j+1})/2)`
lies inside the mission boundary polygon B.

### Point-in-Polygon Test

Membership `c ∈ B` is decided by `GeometryOps.point_in_polygon(c, B)`, which
uses the **ray-casting algorithm**:

```
Cast a horizontal ray from c to (+∞, c_y)
Count the number of boundary edges it crosses
c ∈ B  ⟺  crossing count is odd
```

This runs in O(|B|) per point and is robust to non-convex polygons.

### Cell Area

For a square grid cell with resolution r:

```
area = r²   (m²)
```

---

## 3  Cell Priority Scoring

Each cell is assigned a priority `π_c ∈ [0.1, 1.0]` that determines the order
in which cells are visited.  Higher priority → visited earlier in the mission.

### Gaussian Distance-from-Centre Priority

The priority decreases exponentially with distance from the area centre:

```
area_center = mean(B)                  (centroid of boundary polygon)
dist_center = ‖c − area_center‖       (cell-to-centre distance)
area_scale  = ‖max(B) − min(B)‖       (diagonal of bounding box)

π_c = exp(−dist_center / (area_scale / 10))
π_c = clip(π_c, 0.1, 1.0)
```

**Rationale:** ISR missions typically originate from a headquarters position
near the area centre; central regions are higher-value targets.  The `/10`
factor controls the priority falloff radius.

---

## 4  Cell-to-Drone Assignment

Cells are assigned to drones using **greedy nearest-neighbour assignment**,
applied in priority-sorted order:

```
1. Sort cells by priority (descending)
2. For each cell c (in priority order):
     j* = argmin_{j=1,…,n} ‖c.center − p_j[:2]‖
     Assign c → drone j*
```

This is a **greedy spatial partition**: each cell goes to the nearest drone,
so drones naturally cover geographically close regions.  It does not guarantee
equal workload distribution, but it produces short individual paths.

**Complexity:** O(m log m) for sorting + O(m n) for assignment,
where m = number of cells and n = number of drones.

---

## 5  Waypoint Generation

After assignment, each drone d receives a list of cells `{c₁, c₂, …, c_k}`.
The waypoint generator orders these cells and returns a 3-D waypoint array
`W ∈ ℝᵏˣ³`:

```
W[i] = [c_i.center[0],  c_i.center[1],  altitude]
```

Three path strategies are implemented:

### 5.1  Nearest-Neighbour Path

A **greedy tour** starting from the drone's current position:

```
remaining = {c₁, …, c_k}
path = []
current = drone_start_position

While remaining ≠ ∅:
    c* = argmin_{c ∈ remaining} ‖c.center − current‖
    path.append(c*)
    current = c*.center
    remaining.remove(c*)
```

This is the classic **nearest-neighbour heuristic** for the Travelling Salesman
Problem (TSP).  It is not optimal in general, but runs in O(k²) and typically
gives tours within 20–25% of the optimum.

### 5.2  Sweep (Boustrophedon) Path

Cells are sorted by y-coordinate first, then x within each y-band:

```
path = sorted(cells, key = lambda c: (c.center[1], c.center[0]))
```

The resulting path resembles a **lawnmower** (boustrophedon) pattern: the drone
sweeps left-to-right across one row, then right-to-left across the next.  This
minimises backtracking for rectangular mission areas.

### 5.3  Spiral Path

Cells are sorted by distance from the centroid of all assigned cells,
from nearest to farthest:

```
centroid = mean({c.center for c in cells})
path = sorted(cells, key = lambda c: ‖c.center − centroid‖)
```

This visits the highest-priority (central) cells first and spirals outward —
useful when early coverage of the centre is critical.

---

## 6  Mission Time Estimation

Given a waypoint sequence `W = [w₀, w₁, …, w_{k-1}]` with hover time `h`
per waypoint and drone cruise speed `v`:

```
path_length = Σ_{i=0}^{k-2} ‖w_{i+1} − w_i‖    (total path length)
travel_time = path_length / v                      (travel component)
hover_time  = k × h                                (dwell at each waypoint)

total_time  = (travel_time + hover_time) × 1.2    (20% buffer)
```

The 20 % buffer accounts for turns, altitude changes, and acceleration phases
not captured by the constant-speed model.

**Edge case:** If `k = 1`, there is no travel (only hover):

```
total_time = hover_time × 1.2
```

---

## 7  Coverage Tracking

A **boolean coverage matrix** tracks which cells have been visited:

```
coverage_matrix ∈ {False, True}^m
```

A cell is marked as covered when the drone passes within `ρ_s` metres of its
centroid (sensor footprint check).  The coverage ratio is:

```
coverage_ratio = sum(coverage_matrix) / m  ∈ [0, 1]
```

The mission continues until `coverage_ratio ≥ coverage_goal` (default 0.95)
or the maximum mission duration is reached.

---

## 8  Complexity Analysis

| Operation | Complexity |
|-----------|-----------|
| Grid decomposition | O(m_total × |B|) where m_total = (x_range/r) × (y_range/r) |
| Point-in-polygon per cell | O(|B|) |
| Priority scoring | O(m) |
| Cell-to-drone assignment | O(m log m + m n) |
| Nearest-neighbour tour | O(k²) per drone |
| Sweep / spiral tour | O(k log k) per drone |

For a 500×500 m area with r = 20 m and n = 4 drones:
- m ≈ 625 cells, k ≈ 156 cells per drone
- Nearest-neighbour tour: ~24,000 distance computations per drone — negligible
  on modern hardware (< 1 ms).
