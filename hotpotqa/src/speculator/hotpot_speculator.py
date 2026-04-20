import re
import time
from typing import List

from .api import SpeculatorAPI
from .types import ActionPrediction, FeedbackRecord, ObservationPrediction
from .. import constants
from ..llm_client import LLMClient
from ..prompts import PromptTemplates


class HotPotQASpeculator(SpeculatorAPI):
    """Speculator implementation for HotPotQA.

    It predicts both next actions and search observations while maintaining
    an internal speculative state for lookup steps.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.action_llm = LLMClient(
            model_name=model_name,
            temperature=constants.guess_temperature,
            max_tokens=constants.max_guess_output_tokens,
            top_p=constants.guess_top_p,
        )
        self.observation_llm = self.action_llm
        self.feedback: List[FeedbackRecord] = []
        self.reset_episode()

    def reset_episode(self) -> None:
        self.page = None
        self.obs = ""
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None

    @staticmethod
    def _extract_actions(action_string: str) -> List[str]:
        search_pattern = r"[Ss]earch\[[^\]]+\]"
        lookup_pattern = r"[Ll]ookup\[[^\]]+\]"
        finish_pattern = r"[Ff]inish\[[^\]]+\]"
        outputs: List[str] = []
        for pattern in (search_pattern, lookup_pattern, finish_pattern):
            outputs.extend(re.findall(pattern, action_string))
        return outputs

    @staticmethod
    def _separate_thought_and_actions(step_index: int, thought_action: str):
        try:
            split = thought_action.split(f"Thought {step_index}: ")
            thought = split[0].strip() if len(split) == 1 else split[1].strip()
        except IndexError:
            thought = thought_action.strip()
        actions = HotPotQASpeculator._extract_actions(thought_action)
        return thought, actions

    @staticmethod
    def _get_page_obs(page: str) -> str:
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for paragraph in paragraphs:
            sentences += paragraph.split('. ')
        sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]
        return ' '.join(sentences[:5])

    def _construct_lookup_list(self, keyword: str):
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for paragraph in paragraphs:
            sentences += paragraph.split('. ')
        sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]
        return [sentence for sentence in sentences if keyword.lower() in sentence.lower()]

    def predict_actions(
        self,
        step_index: int,
        running_prompt: str,
        num_actions: int,
        max_retries: int,
    ) -> ActionPrediction:
        n_calls = 1
        n_badcalls = 0

        thought_action = self.action_llm.call(
            running_prompt + PromptTemplates.ACTION_GUESS_PROMPT.format(i=step_index, num_guesses=num_actions),
            stop=None,
        )
        thought, actions = self._separate_thought_and_actions(step_index, thought_action)

        retry_attempt = 1
        while not actions and retry_attempt <= max_retries:
            n_calls += 1
            n_badcalls += 1
            retry_response = self.action_llm.call(
                running_prompt + PromptTemplates.RETRY_PROMPT.format(
                    attempt=retry_attempt,
                    role=constants.agent_role,
                    num_guesses=num_actions,
                )
            )
            actions = self._extract_actions(retry_response)
            retry_attempt += 1

        if not actions:
            raise ValueError("Action not found in LLM output after all retries")

        return ActionPrediction(
            thought=thought,
            actions=actions,
            n_calls=n_calls,
            n_badcalls=n_badcalls,
            raw_response=thought_action,
        )

    def _predict_search_observation(self, entity: str, max_retries: int):
        prompt = PromptTemplates.GUESS_STEP_PROMPT.format(entity)
        llm_response = None
        for _ in range(max_retries):
            llm_response = self.observation_llm.call(prompt)
            if llm_response:
                break
        if not llm_response:
            llm_response = ""

        self.page = llm_response
        self.obs = self._get_page_obs(llm_response)
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None

        return self.obs, llm_response

    def _predict_lookup_observation(self, keyword: str):
        if self.lookup_keyword != keyword:
            self.lookup_keyword = keyword
            self.lookup_list = self._construct_lookup_list(keyword)
            self.lookup_cnt = 0

        if self.lookup_cnt >= len(self.lookup_list):
            self.obs = "No more results."
        else:
            self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) {self.lookup_list[self.lookup_cnt]}"
            self.lookup_cnt += 1

        return self.obs

    def predict_observation(self, action: str, max_retries: int) -> ObservationPrediction:
        normalized_action = action.strip()
        start = time.perf_counter()

        if normalized_action.startswith("search[") and normalized_action.endswith("]"):
            entity = normalized_action[len("search["):-1]
            observation, raw_page = self._predict_search_observation(entity, max_retries=max_retries)
        elif normalized_action.startswith("lookup[") and normalized_action.endswith("]"):
            keyword = normalized_action[len("lookup["):-1]
            observation = self._predict_lookup_observation(keyword)
            raw_page = None
        elif normalized_action.startswith("finish[") and normalized_action.endswith("]"):
            observation = "Episode finished, reward = 0"
            raw_page = None
        elif normalized_action.startswith("think[") and normalized_action.endswith("]"):
            observation = "Nice thought."
            raw_page = None
        else:
            observation = f"Invalid action: {normalized_action}"
            raw_page = None

        latency = time.perf_counter() - start
        return ObservationPrediction(
            observation=observation,
            latency_s=latency,
            source_action=normalized_action,
            raw_page=raw_page,
        )

    def record_feedback(
        self,
        action: str,
        real_observation: str,
        predicted_observation: str,
    ) -> None:
        self.feedback.append(
            FeedbackRecord(
                action=action,
                real_observation=real_observation,
                predicted_observation=predicted_observation,
            )
        )
