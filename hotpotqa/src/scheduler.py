from collections import defaultdict, deque


class CostAwareScheduler:
    """Minimal cost-aware scheduler for stage 3.

    Uses smoothed hit-rate + latency window to compute:
      expected_benefit = p/(1+p) * api_time/(api_time + spec_time)
    and decides whether to speculate by thresholding expected_benefit.
    """

    def __init__(
            self,
            threshold=0.05,
            window_size=50,
            beta_prior_a=1.0,
            beta_prior_b=1.0,
            default_api_time=1.0,
            default_spec_time=0.1,
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.beta_prior_a = beta_prior_a
        self.beta_prior_b = beta_prior_b
        self.default_api_time = default_api_time
        self.default_spec_time = default_spec_time

        self.hit_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.api_latency_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.spec_latency_history = defaultdict(lambda: deque(maxlen=self.window_size))

    @staticmethod
    def is_eligible(action_type):
        return action_type == "search"

    @staticmethod
    def _avg(values, default_value):
        if not values:
            return default_value
        return sum(values) / len(values)

    def _smoothed_p(self, action_type):
        hits = list(self.hit_history[action_type])
        n = len(hits)
        s = sum(hits)
        return (s + self.beta_prior_a) / (n + self.beta_prior_a + self.beta_prior_b)

    def _benefit_components(self, action_type):
        p_hat = self._smoothed_p(action_type)
        api_time = self._avg(self.api_latency_history[action_type], self.default_api_time)
        spec_time = self._avg(self.spec_latency_history[action_type], self.default_spec_time)
        if api_time <= 0:
            api_time = self.default_api_time
        if spec_time <= 0:
            spec_time = self.default_spec_time
        window_util = api_time / (api_time + spec_time)
        expected_benefit = (p_hat / (1 + p_hat)) * window_util
        return expected_benefit, p_hat, api_time, spec_time, window_util

    def should_speculate(self, action_type):
        if not self.is_eligible(action_type):
            return False, "not_eligible", None

        expected_benefit, p_hat, api_time, spec_time, window_util = self._benefit_components(action_type)
        if expected_benefit > self.threshold:
            return True, None, expected_benefit
        return False, "low_benefit", expected_benefit

    def record_step(self, action_type, tool_latency, speculated, hit=None, speculator_latency=None):
        if not self.is_eligible(action_type):
            return

        if tool_latency is not None:
            self.api_latency_history[action_type].append(tool_latency)

        if speculated and speculator_latency is not None:
            self.spec_latency_history[action_type].append(speculator_latency)

        if speculated and hit is not None:
            self.hit_history[action_type].append(1 if hit else 0)
