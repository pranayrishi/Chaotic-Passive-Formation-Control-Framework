"""Drag plate deployment scheduler with physical constraints."""
import numpy as np


class DragPlateScheduler:
    """
    Manages binary drag plate ON/OFF timing.
    Physical constraints from S-NET operational experience.
    """

    def __init__(self, min_dwell=30.0, max_switch_rate=1/60.0):
        self.min_dwell = min_dwell  # [s]
        self.max_switch_rate = max_switch_rate  # [Hz]
        self.last_switch_time = -np.inf
        self.current_state = 0  # 0=stowed, 1=deployed
        self.switch_log = []
        self.total_deployed_time = 0.0
        self._last_update_time = 0.0

    def request_switch(self, t_current, requested_state, priority=0):
        """Request state change. Enforces physical constraints."""
        # Track deployed time
        if self.current_state == 1:
            self.total_deployed_time += max(0, t_current - self._last_update_time)
        self._last_update_time = t_current

        if t_current - self.last_switch_time < self.min_dwell:
            return self.current_state

        if requested_state != self.current_state:
            self.current_state = int(requested_state)
            self.last_switch_time = t_current
            self.switch_log.append({
                'time': t_current,
                'state': self.current_state,
                'priority': priority
            })

        return self.current_state

    def get_duty_cycle(self, t_current):
        """Compute current duty cycle (fraction of time deployed)."""
        if t_current <= 0:
            return 0.0
        total = self.total_deployed_time
        if self.current_state == 1:
            total += max(0, t_current - self._last_update_time)
        return total / t_current

    def get_switch_count(self):
        """Total number of switches."""
        return len(self.switch_log)

    def get_switch_times(self):
        """Array of switch times."""
        return np.array([s['time'] for s in self.switch_log]) if self.switch_log else np.array([])

    def reset(self):
        """Reset scheduler state."""
        self.last_switch_time = -np.inf
        self.current_state = 0
        self.switch_log = []
        self.total_deployed_time = 0.0
        self._last_update_time = 0.0
