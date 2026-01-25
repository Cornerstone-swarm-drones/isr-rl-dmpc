"""
Visualization utilities for ISR-RL-DMPC.

Provides plotting and rendering functions for mission visualization,
trajectory analysis, and learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import matplotlib.animation as animation


class TrajectoryVisualizer:
    """Visualize drone trajectories and swarm behavior."""

    @staticmethod
    def plot_2d_trajectory(trajectories: Dict[int, np.ndarray], figsize=(12, 10),
                           title: str = "Drone Trajectories (Top View)") -> plt.Figure:
        """
        Plot 2D drone trajectories (x-y plane).

        Args:
            trajectories: Dict mapping drone_id -> Nx3 position array
            figsize: Figure size tuple
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

        for (drone_id, traj), color in zip(trajectories.items(), colors):
            positions = traj[:, :2]  # Extract x, y
            ax.plot(positions[:, 0], positions[:, 1], 'o-', label=f'Drone {drone_id}',
                    color=color, markersize=3, linewidth=1.5)

            # Start point (green) and end point (red)
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start' if drone_id == 0 else '')
            ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=8, label='End' if drone_id == 0 else '')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

        return fig

    @staticmethod
    def plot_3d_trajectory(trajectories: Dict[int, np.ndarray], figsize=(12, 10),
                           title: str = "3D Drone Trajectories") -> plt.Figure:
        """
        Plot 3D drone trajectories.

        Args:
            trajectories: Dict mapping drone_id -> Nx3 position array
            figsize: Figure size tuple
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

        for (drone_id, traj), color in zip(trajectories.items(), colors):
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', label=f'Drone {drone_id}',
                    color=color, linewidth=1.5, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker='o', s=100,
                      color=color, edgecolor='black', linewidth=2)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], marker='s', s=100,
                      color=color, edgecolor='red', linewidth=2)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()

        return fig

    @staticmethod
    def plot_velocity_profile(trajectories: Dict[int, np.ndarray], dt: float = 0.02,
                             figsize=(12, 6)) -> plt.Figure:
        """
        Plot velocity magnitude over time for all drones.

        Args:
            trajectories: Dict mapping drone_id -> Nx3 position array
            dt: Timestep between positions (seconds)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for drone_id, traj in trajectories.items():
            # Compute velocities from position differences
            velocities = np.diff(traj, axis=0) / dt
            speeds = np.linalg.norm(velocities, axis=1)
            time = np.arange(len(speeds)) * dt

            ax.plot(time, speeds, label=f'Drone {drone_id}', linewidth=2, alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Velocity Magnitude vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    @staticmethod
    def plot_altitude_profile(trajectories: Dict[int, np.ndarray], dt: float = 0.02,
                             figsize=(12, 6)) -> plt.Figure:
        """
        Plot altitude (z) over time.

        Args:
            trajectories: Dict mapping drone_id -> Nx3 position array
            dt: Timestep
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for drone_id, traj in trajectories.items():
            time = np.arange(len(traj)) * dt
            ax.plot(time, traj[:, 2], label=f'Drone {drone_id}', linewidth=2, alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Altitude vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig


class MissionVisualizer:
    """Visualize mission planning and coverage."""

    @staticmethod
    def plot_grid_coverage(area_boundary: np.ndarray, grid_cells: List[np.ndarray],
                          coverage_status: np.ndarray, figsize=(10, 10)) -> plt.Figure:
        """
        Plot grid coverage with covered/uncovered cells.

        Args:
            area_boundary: Nx2 polygon vertices of mission area
            grid_cells: List of Mx2 arrays, each is a cell (quad or triangle)
            coverage_status: Boolean array indicating coverage
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot area boundary
        boundary_closed = np.vstack([area_boundary, area_boundary[0]])
        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', linewidth=2, label='Area Boundary')

        # Plot coverage cells
        for i, cell in enumerate(grid_cells):
            color = 'lightgreen' if coverage_status[i] else 'lightcoral'
            edgecolor = 'darkgreen' if coverage_status[i] else 'darkred'

            poly = Polygon(cell, closed=True, facecolor=color, edgecolor=edgecolor,
                          linewidth=1, alpha=0.6)
            ax.add_patch(poly)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        coverage_pct = 100 * np.sum(coverage_status) / len(coverage_status)
        ax.set_title(f'Mission Area Coverage: {coverage_pct:.1f}%')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Covered'),
            Patch(facecolor='lightcoral', edgecolor='darkred', label='Uncovered')
        ]
        ax.legend(handles=legend_elements)

        return fig

    @staticmethod
    def plot_waypoint_plan(waypoints: np.ndarray, figsize=(10, 10),
                          title: str = "Mission Waypoint Plan") -> plt.Figure:
        """
        Plot waypoint sequence for mission.

        Args:
            waypoints: Nx2 or Nx3 array of waypoint positions
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if waypoints.shape[1] >= 2:
            # Plot waypoint path
            ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=1, alpha=0.5)

            # Plot waypoints with numbers
            for i, wp in enumerate(waypoints):
                ax.plot(wp[0], wp[1], 'ro', markersize=8)
                ax.text(wp[0], wp[1], f'  {i}', fontsize=8, ha='left')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        return fig


class LearningVisualizer:
    """Visualize learning progress and metrics."""

    @staticmethod
    def plot_training_curves(metrics: Dict[str, List[float]], figsize=(14, 6),
                            smooth: bool = True, window_size: int = 10) -> plt.Figure:
        """
        Plot training curves for multiple metrics.

        Args:
            metrics: Dict mapping metric_name -> list of values
            figsize: Figure size
            smooth: Whether to smooth curves
            window_size: Smoothing window size

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            if smooth and len(values) > window_size:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(values, size=window_size)
                ax.plot(smoothed, linewidth=2, label='Smoothed')
                ax.plot(values, alpha=0.3, label='Raw')
            else:
                ax.plot(values, linewidth=2)

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Episode')
            ax.grid(True, alpha=0.3)
            if smooth and len(values) > window_size:
                ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_reward_components(reward_dict: Dict[str, List[float]], figsize=(12, 6)) -> plt.Figure:
        """
        Plot individual reward components over training.

        Args:
            reward_dict: Dict mapping component_name -> list of values
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set3(np.linspace(0, 1, len(reward_dict)))

        for (component, values), color in zip(reward_dict.items(), colors):
            ax.plot(values, label=component, linewidth=2, color=color, alpha=0.7)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Contribution')
        ax.set_title('Reward Component Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    @staticmethod
    def plot_value_function_heatmap(value_grid: np.ndarray, figsize=(10, 8),
                                    title: str = "Value Function") -> plt.Figure:
        """
        Plot 2D heatmap of value function.

        Args:
            value_grid: 2D array of value estimates
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(value_grid, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xlabel('X State Dimension')
        ax.set_ylabel('Y State Dimension')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value Estimate')

        return fig


class FormationVisualizer:
    """Visualize drone formations and swarm configurations."""

    @staticmethod
    def plot_formation_snapshot(positions: np.ndarray, comm_radius: float,
                               figsize=(10, 10), title: str = "Swarm Formation") -> plt.Figure:
        """
        Plot snapshot of drone formation with communication links.

        Args:
            positions: Nx3 array of drone positions
            comm_radius: Communication radius (visualization only)
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot communication links
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i, :2] - positions[j, :2])
                if dist <= comm_radius:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           'g--', alpha=0.3, linewidth=1)

        # Plot drones
        ax.scatter(positions[:, 0], positions[:, 1], s=200, c='blue', marker='o',
                  edgecolors='black', linewidth=2, zorder=5, label='Drones')

        # Plot communication range circles
        for i, pos in enumerate(positions):
            circle = Circle((pos[0], pos[1]), comm_radius, fill=False,
                           edgecolor='blue', linestyle=':', alpha=0.3)
            ax.add_patch(circle)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()

        return fig

    @staticmethod
    def plot_inter_drone_distances(positions: np.ndarray, figsize=(10, 8)) -> plt.Figure:
        """
        Plot pairwise inter-drone distance matrix.

        Args:
            positions: Nx3 array of drone positions
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        n = len(positions)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(positions[i] - positions[j])

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(distances, cmap='hot', aspect='auto')

        ax.set_xlabel('Drone j')
        ax.set_ylabel('Drone i')
        ax.set_title('Inter-Drone Distance Matrix')
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Distance (m)')

        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{distances[i, j]:.1f}',
                             ha="center", va="center", color="white", fontsize=8)

        return fig


class EnergyVisualizer:
    """Visualize battery and energy metrics."""

    @staticmethod
    def plot_battery_levels(battery_history: Dict[int, List[float]], dt: float = 0.02,
                           figsize=(12, 6)) -> plt.Figure:
        """
        Plot battery level over time for all drones.

        Args:
            battery_history: Dict mapping drone_id -> list of battery levels (Wh)
            dt: Timestep
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab20(np.linspace(0, 1, len(battery_history)))

        for (drone_id, battery), color in zip(battery_history.items(), colors):
            time = np.arange(len(battery)) * dt
            ax.plot(time, battery, label=f'Drone {drone_id}', linewidth=2,
                   color=color, alpha=0.7)

        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Depleted')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Battery Energy (Wh)')
        ax.set_title('Battery Level vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        return fig

    @staticmethod
    def plot_energy_consumption(energy_consumed: Dict[int, List[float]],
                               figsize=(12, 6)) -> plt.Figure:
        """
        Plot cumulative energy consumption.

        Args:
            energy_consumed: Dict mapping drone_id -> list of cumulative energy (Wh)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab20(np.linspace(0, 1, len(energy_consumed)))

        for (drone_id, energy), color in zip(energy_consumed.items(), colors):
            ax.plot(energy, label=f'Drone {drone_id}', linewidth=2,
                   color=color, alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Energy Consumed (Wh)')
        ax.set_title('Energy Consumption vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig


class StatisticsVisualizer:
    """Visualize statistical summaries."""

    @staticmethod
    def plot_histogram(data: np.ndarray, bins: int = 30, figsize=(10, 6),
                      title: str = "Distribution", xlabel: str = "Value") -> plt.Figure:
        """
        Plot histogram of data.

        Args:
            data: 1D array of values
            bins: Number of bins
            figsize: Figure size
            title: Plot title
            xlabel: X-axis label

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean = np.mean(data)
        std = np.std(data)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2, label=f'±1σ: {std:.2f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2)
        ax.legend()

        return fig

    @staticmethod
    def plot_correlation_matrix(data: np.ndarray, variable_names: List[str],
                               figsize=(10, 8)) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            data: NxM array where N is samples and M is variables
            variable_names: List of M variable names
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        corr_matrix = np.corrcoef(data.T)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(np.arange(len(variable_names)))
        ax.set_yticks(np.arange(len(variable_names)))
        ax.set_xticklabels(variable_names, rotation=45, ha='right')
        ax.set_yticklabels(variable_names)

        ax.set_title('Correlation Matrix')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        # Add text annotations
        for i in range(len(variable_names)):
            for j in range(len(variable_names)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                             fontsize=9)

        plt.tight_layout()
        return fig
