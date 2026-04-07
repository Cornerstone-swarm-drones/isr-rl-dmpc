# Task Allocation — Bipartite Assignment and the Hungarian Algorithm

**Source file:**
- `src/isr_rl_dmpc/modules/task_allocator.py` — `HungarianAssignment`, `TaskAllocator`

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Cost Matrix Construction](#2-cost-matrix-construction)
3. [The Assignment Problem](#3-the-assignment-problem)
4. [Hungarian Algorithm](#4-hungarian-algorithm)
5. [Rectangular Matrices and Padding](#5-rectangular-matrices-and-padding)
6. [Greedy Fallback](#6-greedy-fallback)
7. [Complexity Analysis](#7-complexity-analysis)
8. [References](#8-references)

---

## 1  Problem Formulation

Given:
- **m** ISR tasks `{τ₁, …, τₘ}` (detect, classify, track, cover, transit)
- **n** available drones `{d₁, …, dₙ}`
- A cost matrix **C ∈ ℝᵐˣⁿ** where `C[i, j]` is the cost of assigning task i to drone j

Find a one-to-one assignment `σ: tasks → drones` that minimises total cost:

$$
\min_{\sigma} \; \sum_i C[i,\, \sigma(i)]
\quad \text{s.t.} \quad \sigma \text{ is a bijection (each task to exactly one drone)}
$$

This is the **Linear Assignment Problem (LAP)** — a combinatorial optimisation
problem on a bipartite graph.

---

## 2  Cost Matrix Construction

The cost of assigning task $\tau_i$ to drone $d_j$ is a weighted sum of five factors:

$$
C[i,j] = 0.30\,c_{\text{dist}} + 0.20\,c_{\text{fuel}} + 0.20\,c_{\text{load}} + 0.15\,c_{\text{sensor}} - 0.15\,b_{\text{priority}}
$$

### Distance Cost

$$
d_{ij} = \|\tau_i.\mathbf{p} - d_j.\mathbf{p}\|, \qquad
c_{\text{dist}} = d_{ij} / 1000 \quad \text{(normalised to } {\approx}\,[0,2] \text{ for 1 km range)}
$$

### Fuel Cost

$$
c_{\text{fuel}} = (100 - d_j.\text{fuel}) / 100 \in [0, 1]
$$

Drones with more fuel are preferred.

### Task Load Cost

$$
c_{\text{load}} = d_j.\text{current\_load} \in [0, 1]
$$

### Sensor Capability Cost

$$
c_{\text{sensor}} = 1 - \frac{|\tau_i.\text{required} \cap d_j.\text{sensors}|}{|\tau_i.\text{required}|} \in [0, 1]
$$

Drones that have all required sensors get $c_{\text{sensor}} = 0$.

### Priority Bonus

$$
b_{\text{priority}} = \tau_i.\text{priority} \in [0, 1]
$$

### Feasibility Check

If the travel time plus task duration exceeds the drone's endurance, the
assignment is marked **infeasible**:

$$
t_{\text{travel}} = d_{ij} / d_j.v_{\max}, \qquad
t_{\text{total}} = t_{\text{travel}} + \tau_i.\text{duration}
$$

$$
\text{if } t_{\text{total}} > d_j.\text{endurance} \Rightarrow C[i,j] = 10^6 \quad \text{(infeasible)}
$$

---

## 3  The Assignment Problem

The cost matrix $C$ defines a **weighted bipartite graph**:
- Left nodes: tasks
- Right nodes: drones
- Edge weight: cost $C[i, j]$

A **perfect matching** is a set of edges in which every task and every drone
appears exactly once.  The minimum-cost perfect matching is the optimal
assignment.

For a square $n \times n$ matrix, there are $n!$ possible matchings; exhaustive
search is infeasible for $n > 12$.  The **Hungarian algorithm** finds the
optimal matching in $O(n^3)$ time.

---

## 4  Hungarian Algorithm

The implementation follows the **potential-augmenting-path** (Jonker-Volgenant)
variant operating on the cost matrix directly.

### Dual Variables (Potentials)

The algorithm maintains dual variables `u[i]` (row potentials) and `v[j]`
(column potentials) such that:

```
u[i] + v[j] ≤ C[i, j]   for all (i, j)
u[i] + v[j] = C[i, j]   for matched pairs (i, j)
```

This is the LP dual of the assignment problem.  By strong duality, the
primal and dual optima coincide when a perfect matching with zero reduced
cost exists.

**Reduced cost:** `C̃[i, j] = C[i, j] − u[i] − v[j] ≥ 0`

### Algorithm Steps (Pseudo-Code)

```
Initialise u = 0, v = 0, p[j] = 0 (column assignment), way[j] = 0

For each task i = 1, …, n:
  1. p[0] = i                          # "virtual" source task
  2. Set j₀ = 0 (unmatched column)
  3. minv[j] = ∞, used[j] = false for all j

  Repeat (Dijkstra-like shortest-path):
    a. Mark j₀ as used
    b. i₀ = p[j₀]                     # task currently assigned to column j₀
    c. Find j₁ = argmin_{unused j} C̃[i₀, j]  and update minv, way
    d. delta = minv[j₁]
    e. Update potentials: u[p[j]] += delta (used j); v[j] -= delta (used j);
                          minv[j] -= delta (unused j)
    f. j₀ = j₁
  Until p[j₀] == 0 (reached unmatched column)

  Augment the matching by reversing the way[] path from j₀ back to 0.
```

### Augmentation

The `way[]` array records the predecessor column in the shortest-path tree.
Augmentation follows the alternating path:

```
j₀ → way[j₀] → way[way[j₀]] → … → 0
```

reversing matched/unmatched edges to extend the matching by one pair.

### Correctness

By the **König–Egerváry theorem**, the minimum-weight perfect matching equals
the maximum dual objective.  The algorithm terminates with an optimal
assignment because:
1. Potentials are always feasible (reduced costs ≥ 0).
2. Augmentation strictly increases the size of the matching.
3. After n augmentations, the matching is perfect.

---

## 5  Rectangular Matrices and Padding

When the number of tasks `m ≠ n` (drones), the matrix is padded to a square
`max(m, n) × max(m, n)` matrix with infeasible cost entries (1e6):

```
m > n  (more tasks than drones):
    C_padded = [C | 1e6 ones(m, m−n)]    (add dummy drones)

m < n  (more drones than tasks):
    C_padded = [C ; 1e6 ones(n−m, n)]    (add dummy tasks)
```

After solving, only the original `m` assignments are returned.  Drones
assigned to dummy tasks are considered **unassigned**.

---

## 6  Greedy Fallback

When there are more tasks than drones, all tasks cannot be assigned
simultaneously.  The **greedy fallback** prioritises tasks by `τᵢ.priority`
(descending) and assigns each to the lowest-cost available drone:

```
Sort tasks by priority (descending)
For each task τᵢ in order:
    j* = argmin_{unassigned j} C[i, j]
    Assign τᵢ → d_{j*}
    Mark j* as assigned
```

This greedy scheme is not optimal (it ignores future task–drone interactions)
but runs in O(m n) time and handles real-time task surges.

---

## 7  Complexity Analysis

| Variant | Time Complexity | Use Case |
| :--- | :--- | :--- |
| Hungarian (square n×n) | O(n³) | m ≤ n (≤ drones) |
| Greedy (m > n) | O(m n) | More tasks than drones |

For a typical swarm of n = 6 drones and m = 10 tasks, the Hungarian algorithm
runs in microseconds on modern hardware.

---

## 8  References

1. H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval
   Research Logistics Quarterly*, 2(1–2):83–97, 1955.
2. J. Munkres, "Algorithms for the assignment and transportation problems,"
   *Journal of the Society for Industrial and Applied Mathematics*,
   5(1):32–38, 1957.
3. R. Jonker and A. Volgenant, "A shortest augmenting path algorithm for
   dense and sparse linear assignment problems," *Computing*, 38:325–340, 1987.
