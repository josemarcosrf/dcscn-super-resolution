import numpy as np

from typing import Dict, List

Metrics = Dict[str, List[float]]


class MetricTracker():
    def __init__(self, metric_name: str, checking_func,
                 tracking_criteria: Metrics):
        """Checks if a condition is met on a sequence of measurements.

        Args:
            metric_name (string): name of the metric to track
            checking_func (function): boolean function checking if a specific
                                      criteria has been met on a
                                      sequence of values.
            tracking_criteria (dict): additional configuration needed
                                      for the condition checking function
        """
        self.metric_name = metric_name
        self.checking_func = checking_func
        self.tracking_criteria = tracking_criteria

        # basic stats from the seen data
        self.history_length = 0

    def check(self, val_metrics: Metrics) -> bool:
        measurements = val_metrics[self.metric_name]

        # perform a check only if new data is available
        if len(measurements) > self.history_length:
            self.history_length = len(measurements)
            return self.checking_func(measurements,
                                      **self.tracking_criteria)

    @classmethod
    def is_increasing(cls, ctrl_measures: List[float],
                      patience: float) -> bool:
        """Checks if a sequence of given metrics is extrictly increasing
        in the last 'patience' time-steps.

        Args:
            ctrl_measures (list or np.array): array of numeric measures
            patience (int): number of time-steps.

        Returns:
            bool: whether the measure is increasing or not
        """
        if len(ctrl_measures) >= patience:
            return all([a > b
                        for a, b in zip(
                            ctrl_measures[:-1],
                            ctrl_measures[1:])])
        return False

    @classmethod
    def is_decreasing(cls, ctrl_measures: List[float],
                      patience: float) -> bool:
        """Checks if a sequence of given metrics is extrictly decreasing
        in the last 'patience' time-steps.

        Args:
            ctrl_measures (list or np.array): array of numeric measures
            patience (int): number of time-steps.

        Returns:
            bool: whether the measure is increasing or not
        """
        if len(ctrl_measures) >= patience:
            return all([a < b
                        for a, b in zip(
                            ctrl_measures[:-1],
                            ctrl_measures[1:])])
        return False

    @classmethod
    def is_stagnated(cls, ctrl_measures: List[float],
                     patience: int, tolerance: float = 1e-4) -> bool:
        """Check whether a numeric measure sequence has been stagnated
        under a given 'tolerance' for a 'patience' number of steps.

        Args:
            ctrl_measures (list or np.array): array of numeric measures
            patience (int): number of time-steps
            tolerance (float, optional): absolute numeric tolerance to
            consider two measurements are close in magnitud. Defaults to 1e-4.

        Returns:
            bool: whether the measure is stagnated or not
        """
        if len(ctrl_measures) >= patience:
            return all([np.isclose(a, b, atol=tolerance)
                        for a, b in zip(ctrl_measures[:-1],
                                        ctrl_measures[1:])])
        return False

    @classmethod
    def is_max(cls, ctrl_measures: List[float]) -> bool:
        if len(ctrl_measures) == 0:
            return False
        if len(ctrl_measures) == 1:
            return True
        if len(ctrl_measures) >= 2:
            last = ctrl_measures[-1]
            return last > max(ctrl_measures[:-1])
        return False

    @classmethod
    def is_min(cls, ctrl_measures: List[float]) -> bool:
        if len(ctrl_measures) == 0:
            return False
        if len(ctrl_measures) == 1:
            return True
        if len(ctrl_measures) >= 2:
            last = ctrl_measures[-1]
            return last < min(ctrl_measures[:-1])
        return False
