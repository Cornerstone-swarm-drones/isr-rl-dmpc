#!/usr/bin/env python3
"""
Visualize the reduced local POMDP used for patrol-risk vs persistent-threat reasoning.

This script is diagnostic. It does not control the full simulator. Its goal is
to show how a tiny approximate-belief planner behaves in the local decision
regime where a region may be either:

* merely neglected and recoverable by observation, or
* hiding a persistent threat that survives observation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(ROOT / "src"))

from isr_rl_dmpc.core import (
    QMDPPlanner,
    ReducedLocalPatrolPOMDP,
    ReducedPatrolAction,
    ReducedPatrolObservation,
)


ACTION_COLORS = ["#2A9D8F", "#E9C46A", "#E76F51"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the reduced local patrol POMDP and QMDP policy."
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "visualizations" / "reduced_patrol_pomdp.png"),
    )
    parser.add_argument("--resolution", type=int, default=81)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _plot_policy_region(ax: plt.Axes, threat_axis: np.ndarray, neglect_axis: np.ndarray, action_grid: np.ndarray) -> None:
    cmap = ListedColormap(ACTION_COLORS)
    image = ax.imshow(
        action_grid,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    ax.set_title("QMDP Policy Regions")
    ax.set_xlabel("Belief: persistent threat probability")
    ax.set_ylabel("Belief: neglect probability")
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["routine", "revisit", "escalate"])


def _plot_value_heatmap(ax: plt.Axes, value_grid: np.ndarray) -> None:
    image = ax.imshow(
        value_grid,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="nearest",
        cmap="viridis",
        aspect="auto",
    )
    ax.set_title("QMDP Value Surface")
    ax.set_xlabel("Belief: persistent threat probability")
    ax.set_ylabel("Belief: neglect probability")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_belief_update_slice(ax: plt.Axes, pomdp: ReducedLocalPatrolPOMDP) -> None:
    prior_axis = np.linspace(0.0, 1.0, 201)
    fixed_neglect = 0.7

    for observation, color in [
        (ReducedPatrolObservation.QUIET, "#2A9D8F"),
        (ReducedPatrolObservation.ELEVATED, "#E9C46A"),
        (ReducedPatrolObservation.PERSISTENT, "#E76F51"),
    ]:
        posterior_threat = []
        for prior_threat in prior_axis:
            belief = pomdp.factorized_belief(prior_threat, fixed_neglect)
            posterior = pomdp.belief_update(
                belief,
                ReducedPatrolAction.FOCUSED_REVISIT,
                observation,
            )
            posterior_threat.append(
                posterior[2] + posterior[3]
            )
        ax.plot(
            prior_axis,
            posterior_threat,
            linewidth=2.0,
            color=color,
            label=pomdp.observation_labels()[int(observation)],
        )

    ax.set_title("Focused-Revisit Belief Update Slice")
    ax.set_xlabel("Prior threat probability")
    ax.set_ylabel("Posterior threat probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.02,
        f"fixed neglect belief = {fixed_neglect:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _draw_transition_graph(ax: plt.Axes, pomdp: ReducedLocalPatrolPOMDP) -> None:
    action = ReducedPatrolAction.FOCUSED_REVISIT
    transition = pomdp.transition_matrix[int(action)]
    labels = pomdp.state_labels()
    positions = {
        0: (0.20, 0.75),
        1: (0.20, 0.25),
        2: (0.78, 0.75),
        3: (0.78, 0.25),
    }

    ax.set_title("Focused-Revisit Transition Graph")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for state_idx, (x_pos, y_pos) in positions.items():
        ax.scatter([x_pos], [y_pos], s=1200, color="#F1FAEE", edgecolors="#1D3557", linewidths=1.5)
        ax.text(x_pos, y_pos, labels[state_idx], ha="center", va="center", fontsize=10)

    for src in range(transition.shape[0]):
        for dst in range(transition.shape[1]):
            probability = float(transition[src, dst])
            if probability < 0.18:
                continue
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            if src == dst:
                ax.annotate(
                    "",
                    xy=(x0 + 0.05, y0 + 0.03),
                    xytext=(x0 - 0.02, y0 + 0.03),
                    arrowprops={"arrowstyle": "->", "color": "#457B9D", "lw": 1.3},
                )
                ax.text(x0 + 0.02, y0 + 0.09, f"{probability:.2f}", fontsize=8, color="#1D3557")
                continue

            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "#457B9D", "lw": 1.1, "alpha": 0.9},
            )
            ax.text(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                f"{probability:.2f}",
                fontsize=8,
                color="#1D3557",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.85},
            )


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)

    pomdp = ReducedLocalPatrolPOMDP()
    planner = QMDPPlanner(pomdp)
    threat_axis, neglect_axis, action_grid, value_grid = planner.belief_grid(resolution=args.resolution)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    _plot_policy_region(axes[0, 0], threat_axis, neglect_axis, action_grid)
    _plot_value_heatmap(axes[0, 1], value_grid)
    _plot_belief_update_slice(axes[1, 0], pomdp)
    _draw_transition_graph(axes[1, 1], pomdp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)

    key_beliefs = {
        "calm": pomdp.factorized_belief(threat_probability=0.05, neglect_probability=0.10),
        "likely neglect": pomdp.factorized_belief(threat_probability=0.10, neglect_probability=0.80),
        "ambiguous": pomdp.factorized_belief(threat_probability=0.45, neglect_probability=0.60),
        "likely threat": pomdp.factorized_belief(threat_probability=0.80, neglect_probability=0.55),
    }

    print("=" * 72)
    print("Reduced local patrol POMDP")
    print("=" * 72)
    print(f"Output                : {output_path}")
    for label, belief in key_beliefs.items():
        action = planner.select_action(belief)
        action_values = planner.action_values(belief)
        print(
            f"{label:<21}: action={pomdp.action_labels()[int(action)]} "
            f"q={np.round(action_values, 3).tolist()}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
