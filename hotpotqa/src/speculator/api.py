from abc import ABC, abstractmethod

from .types import ActionPrediction, ObservationPrediction


class SpeculatorAPI(ABC):
    @abstractmethod
    def reset_episode(self) -> None:
        """Reset per-episode speculative state."""

    @abstractmethod
    def predict_actions(
        self,
        step_index: int,
        running_prompt: str,
        num_actions: int,
        max_retries: int,
    ) -> ActionPrediction:
        """Predict top-k next actions for the current step."""

    @abstractmethod
    def predict_observation(self, action: str, max_retries: int) -> ObservationPrediction:
        """Predict the observation corresponding to an action."""

    @abstractmethod
    def record_feedback(
        self,
        action: str,
        real_observation: str,
        predicted_observation: str,
    ) -> None:
        """Record post-hoc supervision for later analysis or adaptation."""
