from planning.planner_base import PlannerBase


class DummyPlanner(PlannerBase):
    def plan(self, obs, waypoints):
        return 0, 0
