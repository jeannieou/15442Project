from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ActionPrediction:
    thought: str
    actions: List[str]
    n_calls: int
    n_badcalls: int
    raw_response: Optional[str] = None


@dataclass
class ObservationPrediction:
    observation: str
    latency_s: float
    source_action: str
    raw_page: Optional[str] = None


@dataclass
class FeedbackRecord:
    action: str
    real_observation: str
    predicted_observation: str
