"""Minimal ride-duration model used by the FastAPI/Litestar examples."""

from typing import Optional, Protocol


class Estimator(Protocol):
    """Anything with a scikit-learn-style ``predict`` method."""

    def predict(self, features: list[float]) -> list[float]: ...


class _HeuristicEstimator:
    """Default placeholder estimator: distance / speed + per-passenger overhead.

    Mimics a trained estimator's interface (``predict`` returns a batch-like
    list) so it can be swapped for a real model without touching
    :class:`RideDurationModel`.
    """

    #: Assumed average speed in km per minute (~30 km/h).
    AVG_SPEED_KM_PER_MIN = 0.5
    #: Extra minutes added per passenger (boarding overhead).
    PASSENGER_OVERHEAD_MIN = 0.5

    def predict(self, features: list[float]) -> list[float]:
        distance, passengers = features[0], features[1]
        duration = distance / self.AVG_SPEED_KM_PER_MIN
        duration += passengers * self.PASSENGER_OVERHEAD_MIN
        return [round(duration, 2)]


class RideDurationModel:
    """Estimates trip duration in minutes from ``[distance_km, passengers]``.

    Delegates the actual prediction to an internal estimator (``self._model``).
    By default this is a simple heuristic (:class:`_HeuristicEstimator`); assign
    a real trained model to ``self._model`` to swap it in — it only needs a
    scikit-learn-style ``predict(features) -> [value]`` method.

    An optional ``threshold`` caps the returned duration (useful to clip
    implausible predictions).
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self._model: Estimator = _HeuristicEstimator()
        #: Optional upper bound on the predicted duration (minutes).
        self.threshold = threshold

    def predict(self, features: list[float]) -> float:
        """Predict ride duration in minutes.

        Args:
            features: ``[distance_km, passengers]``.

        Returns:
            Estimated duration in minutes. Clipped to ``self.threshold`` when
            that attribute is set.
        """
        duration = float(self._model.predict(features)[0])
        if self.threshold is not None:
            duration = min(duration, self.threshold)
        return duration
