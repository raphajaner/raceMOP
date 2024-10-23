from abc import ABC
from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class PlannerBase(ABC):
    """Base class for planners."""

    def __init__(self, conf):
        self.conf = conf
        self.drawn_waypoints = []

    def plan(self, state, waypoints):
        """Plan a trajectory. Returns a list of (x, y, v) tuples."""
        raise NotImplementedError()

    def render_waypoints(self, e, waypoints):
        """Render the waypoints e."""
        points = np.vstack((waypoints[:, self.conf.maps.wpt_xind], waypoints[:, self.conf.maps.wpt_yind])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                from pyglet.gl import GL_POINTS
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
