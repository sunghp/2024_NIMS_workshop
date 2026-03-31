"""
Muscle Fatigue Modeling Module

Tracks muscle capacity over multiple measurement sessions and computes
fatigue ratio using spline interpolation for capacity estimation.

Approach:
    1. First session: Estimate total muscle capacity (baseline) from
       cumulative RMS energy across all contraction segments.
    2. Subsequent sessions: Compute usage/capacity ratio as fatigue index.
    3. As sessions accumulate, fit a cubic spline to capacity measurements
       to model how total capacity changes over time (e.g., due to training).
    4. Use the interpolated capacity curve to refine fatigue estimates,
       improving precision with each additional session.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class SessionRecord:
    """Record for a single measurement session."""
    timestamp: float  # session time (e.g., days from first session)
    total_capacity: float  # estimated total muscle capacity (RMS-based)
    usage: float  # measured muscle usage in this session
    fatigue_ratio: float  # usage / capacity at this time point


class FatigueModel:
    """
    Multi-session muscle fatigue model with spline-interpolated capacity tracking.

    The model estimates how muscle capacity evolves over time and uses this
    to compute increasingly precise fatigue ratios.
    """

    def __init__(self):
        self.sessions: list[SessionRecord] = []
        self._spline = None

    def estimate_capacity(self, segment_rms_values: list[float],
                          segment_durations: list[float]) -> float:
        """
        Estimate total muscle capacity from a single session's contraction data.

        Capacity is defined as the sum of (RMS * duration) across all segments,
        representing the total energy output the muscle produced.

        Args:
            segment_rms_values: RMS value for each contraction segment.
            segment_durations: Duration (seconds) for each segment.

        Returns:
            Estimated total capacity (sum of RMS * duration).
        """
        return sum(rms * dur for rms, dur in
                   zip(segment_rms_values, segment_durations))

    def add_session(self, timestamp: float,
                    segment_rms_values: list[float],
                    segment_durations: list[float],
                    usage_rms_values: list[float] | None = None,
                    usage_durations: list[float] | None = None) -> SessionRecord:
        """
        Add a measurement session and compute fatigue ratio.

        For the first session, capacity = total energy output (baseline).
        For subsequent sessions, capacity is estimated from the spline model
        if enough data points exist, otherwise from direct measurement.

        Args:
            timestamp: Time of session (e.g., day number).
            segment_rms_values: RMS values from all contraction segments.
            segment_durations: Duration of each segment (seconds).
            usage_rms_values: If separate from capacity measurement, RMS values
                              for the usage portion. Defaults to same as segment data.
            usage_durations: Durations for usage portion.

        Returns:
            SessionRecord with capacity, usage, and fatigue ratio.
        """
        measured_capacity = self.estimate_capacity(
            segment_rms_values, segment_durations
        )

        if usage_rms_values is not None and usage_durations is not None:
            usage = self.estimate_capacity(usage_rms_values, usage_durations)
        else:
            usage = measured_capacity

        # Determine capacity for fatigue calculation
        if len(self.sessions) >= 2 and self._spline is not None:
            # Use spline-interpolated capacity for better precision
            interpolated_capacity = float(self._spline(timestamp))
            # Ensure capacity is positive
            capacity_for_ratio = max(interpolated_capacity, measured_capacity * 0.5)
        else:
            capacity_for_ratio = measured_capacity

        fatigue_ratio = usage / capacity_for_ratio if capacity_for_ratio > 0 else 0.0

        record = SessionRecord(
            timestamp=timestamp,
            total_capacity=measured_capacity,
            usage=usage,
            fatigue_ratio=fatigue_ratio,
        )
        self.sessions.append(record)

        # Update spline with new data
        self._update_spline()

        return record

    def _update_spline(self):
        """
        Refit cubic spline to all capacity measurements.

        Requires at least 2 data points. With more sessions,
        the spline captures capacity trends (growth from training,
        decline from overuse, etc.) more precisely.
        """
        if len(self.sessions) < 2:
            self._spline = None
            return

        timestamps = np.array([s.timestamp for s in self.sessions])
        capacities = np.array([s.total_capacity for s in self.sessions])

        # Sort by time
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        capacities = capacities[sort_idx]

        self._spline = CubicSpline(timestamps, capacities, bc_type='natural')

    def predict_capacity(self, timestamp: float) -> float | None:
        """
        Predict muscle capacity at a given time using the spline model.

        Args:
            timestamp: Time point to predict.

        Returns:
            Predicted capacity, or None if insufficient data.
        """
        if self._spline is None:
            return None
        return float(self._spline(timestamp))

    def get_fatigue_history(self) -> dict:
        """
        Get all session data as arrays for analysis.

        Returns:
            Dictionary with timestamps, capacities, usages, and fatigue_ratios.
        """
        return {
            'timestamps': [s.timestamp for s in self.sessions],
            'capacities': [s.total_capacity for s in self.sessions],
            'usages': [s.usage for s in self.sessions],
            'fatigue_ratios': [s.fatigue_ratio for s in self.sessions],
        }

    def plot_capacity_trend(self, save_path: str | None = None):
        """
        Plot measured capacity points and the spline interpolation curve.

        Args:
            save_path: If provided, save figure to this path.
        """
        if len(self.sessions) < 2:
            print("Need at least 2 sessions to plot capacity trend.")
            return

        history = self.get_fatigue_history()
        timestamps = np.array(history['timestamps'])
        capacities = np.array(history['capacities'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Capacity trend with spline ---
        ax = axes[0]
        ax.scatter(timestamps, capacities, color='red', s=60, zorder=5,
                   label='Measured capacity')

        if self._spline is not None:
            t_fine = np.linspace(timestamps.min(), timestamps.max(), 200)
            c_fine = self._spline(t_fine)
            ax.plot(t_fine, c_fine, 'b-', linewidth=2,
                    label='Spline interpolation')

        ax.set_xlabel('Session Time')
        ax.set_ylabel('Muscle Capacity (RMS energy)')
        ax.set_title('Muscle Capacity Trend Over Sessions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Fatigue ratio over time ---
        ax = axes[1]
        fatigue_ratios = history['fatigue_ratios']
        ax.plot(timestamps, fatigue_ratios, 'o-', color='purple', linewidth=2)
        ax.set_xlabel('Session Time')
        ax.set_ylabel('Fatigue Ratio (usage / capacity)')
        ax.set_title('Fatigue Ratio Over Sessions')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def summary(self):
        """Print summary of all sessions."""
        print(f"\n{'='*60}")
        print(f"Fatigue Model Summary ({len(self.sessions)} sessions)")
        print(f"{'='*60}")
        for i, s in enumerate(self.sessions):
            spline_cap = ""
            if self._spline is not None and i > 0:
                predicted = self.predict_capacity(s.timestamp)
                spline_cap = f", Spline capacity: {predicted:.2f}"
            print(
                f"  Session {i+1} (t={s.timestamp:.1f}): "
                f"Capacity={s.total_capacity:.2f}, "
                f"Usage={s.usage:.2f}, "
                f"Fatigue={s.fatigue_ratio:.4f}"
                f"{spline_cap}"
            )
