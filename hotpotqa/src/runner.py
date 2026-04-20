import os
import random
import re
import sys
import time
from os.path import join

import requests

from . import constants
from . import environment
from . import wrappers
from .llm_client import LLMClient
from .metrics import Metrics
from .prompts import PromptTemplates
from .scheduler import CostAwareScheduler
from .utils import Utils


class HotPotQARun:
    def __init__(
        self,
        model_name="gemini",
        guess_model_name="gemini",
        to_print_output=True,
        spec_policy="always",
        scheduler_threshold=None,
        output_subdir=None,
    ):
        self.llm = LLMClient(
            model_name=model_name,
            temperature=constants.temperature,
            max_tokens=constants.max_output_tokens,
            top_p=constants.top_p,
        )
        self.model_name = model_name
        self.guess_model_name = guess_model_name
        self.env = self._get_env()
        self.simulation_observations_dict = {}
        self.current_index = None
        self.output_subdir = output_subdir
        self.base_traj_path = self.recalc_base_traj_path()
        self.to_print_output = to_print_output

        self.spec_policy = spec_policy
        self.scheduler_threshold = scheduler_threshold
        valid_policies = {"never", "always", "scheduler"}
        if self.spec_policy not in valid_policies:
            raise ValueError(
                f"Invalid spec_policy '{self.spec_policy}'. Expected one of {sorted(valid_policies)}"
            )
        self.scheduler = None
        if self.spec_policy == "scheduler":
            self.scheduler = CostAwareScheduler(
                threshold=0.05 if self.scheduler_threshold is None else self.scheduler_threshold
            )

    def recalc_base_traj_path(self):
        btp = (
            "./run_metrics/agent_"
            + self.model_name.split("/")[-1]
            + "_top"
            + str(constants.guess_num_actions)
            + "/trajs_"
            + self.guess_model_name.split("/")[-1]
        )
        if self.output_subdir:
            btp = join(btp, self.output_subdir)
        self.base_traj_path = btp
        return btp

    def _get_env(self):
        # Keep compatibility with both WikiEnv() and WikiEnv(guess_model_name=...).
        try:
            env = environment.WikiEnv(guess_model_name=self.guess_model_name)
        except TypeError:
            env = environment.WikiEnv()
        env = wrappers.HotPotQAWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
        env = wrappers.HistoryWrapper(env, obs_format="history")
        return env

    def log(self, *args, save_log=True):
        text = " ".join(str(a) for a in args).strip() + "\n"
        if self.to_print_output:
            try:
                print(text, end="")
            except UnicodeEncodeError:
                safe_text = text.encode("ascii", errors="backslashreplace").decode("ascii")
                sys.stdout.write(safe_text)
        if save_log:
            log_path = join(self.base_traj_path, str(self.current_index), "log.txt")
            Utils.append_file(text, log_path)

    def step(self, env, action, simulate=False):
        if simulate:
            start = time.perf_counter()
            obs, r, done, info = env.step(action, step_type="simulate")
            end = time.perf_counter()
            return obs, r, done, info, end - start

        attempts = 0
        while attempts < 10:
            try:
                start = time.perf_counter()
                obs, r, done, info = env.step(action)
                end = time.perf_counter()
                return obs, r, done, info, end - start
            except requests.exceptions.Timeout:
                attempts += 1

        raise TimeoutError("Action timed out after 10 retries")

    def extract_action(self, action_string):
        search_pattern = r"[Ss]earch\[[^\]]+\]"
        lookup_pattern = r"[Ll]ookup\[[^\]]+\]"
        finish_pattern = r"[Ff]inish\[[^\]]+\]"
        for pattern in [search_pattern, lookup_pattern, finish_pattern]:
            output = re.search(pattern, action_string)
            if output:
                return output.group()
        return None

    def extract_actions(self, action_string):
        search_pattern = r"[Ss]earch\[[^\]]+\]"
        lookup_pattern = r"[Ll]ookup\[[^\]]+\]"
        finish_pattern = r"[Ff]inish\[[^\]]+\]"
        outputs = []
        for pattern in [search_pattern, lookup_pattern, finish_pattern]:
            outputs += re.findall(pattern, action_string)
        return outputs

    def action_lowercase(self, action):
        index = action.find("[")
        if index == -1:
            return action[0].lower() + action[1:]
        return action[:index].lower() + action[index:]

    @staticmethod
    def get_action_type(action):
        action = action.strip().lower()
        ind = action.find("[")
        if ind < 0:
            return None
        return action[:ind]

    def separate_thought_and_action(self, i, thought_action):
        action_condition = f"Action {i}: " in thought_action
        thought_condition = f"Thought {i}: " in thought_action
        thought, action = None, None
        if thought_condition and action_condition:
            thought, action = thought_action.strip().split(f"Action {i}: ")
            thought = thought.strip().split(f"Thought {i}: ")[1]
        elif thought_condition:
            thought = thought_action.strip().split(f"Thought {i}: ")[1]
            action = None
        elif action_condition:
            action = thought_action.strip().split(f"Action {i}: ")[1]
            thought = "Let me do the action " + action
        return thought, action

    def separate_thought_and_actions(self, i, thought_action):
        try:
            split = thought_action.split(f"Thought {i}: ")
            thought = split[0].strip() if len(split) == 1 else split[1].strip()
        except IndexError:
            thought = thought_action.strip()
        actions = self.extract_actions(thought_action)
        return thought, actions

    def generate_thought_actions(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        n_calls_badcalls[0] += 1
        thought_action = self.llm.call(
            running_prompt + PromptTemplates.ACTION_GUESS_PROMPT.format(i=i, num_guesses=num_actions),
            stop=None,
        )
        thought, actions = self.separate_thought_and_actions(i, thought_action)

        retry_attempt = 1
        while not actions and retry_attempt <= max_retries:
            self.log(f"  [Retry {retry_attempt}/{max_retries}] No actions found, retrying...")
            n_calls_badcalls[0] += 1
            n_calls_badcalls[1] += 1
            temp_actions = self.llm.call(
                running_prompt
                + PromptTemplates.RETRY_PROMPT.format(
                    attempt=retry_attempt, role=constants.agent_role, num_guesses=num_actions
                )
            )
            actions = self.extract_actions(temp_actions)
            retry_attempt += 1

        if not actions:
            raise ValueError("Action not found in LLM output after all retries")

        return thought, actions

    def webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        done = False
        running_prompt = prompt
        question = self.env.reset(idx=idx)
        running_prompt += question + "\n"
        self.env.normal_trajectory_dict["prompt"] = running_prompt

        if to_print:
            self.log(f"\n{'=' * 60}")
            self.log(f"  Question [{idx}]: {question}")
            self.log(f"{'=' * 60}")

        sim_running_prompt = running_prompt
        if simulate:
            self.env.sim_trajectory_dict["prompt"] = sim_running_prompt

        n_calls_badcalls = [0, 0]
        step_records = []

        for i in range(1, n):
            if to_print:
                self.log(f"\n--- Step {i}/{n - 1} ---")

            try:
                thought, actions = self.generate_thought_actions(
                    i, running_prompt, n_calls_badcalls, max_retries=constants.max_agent_retries
                )
                action = actions[0]
            except ValueError as e:
                self.log(f"[ERROR] {e}", save_log=False)
                continue

            action_type = self.get_action_type(action)
            eligible_for_speculation = action_type == "search"
            speculated = False
            skip_reason = "policy_disabled"
            expected_benefit = None
            hit = None
            speculator_latency = None
            sim_thought = ""
            sim_actions = []
            sim_obs = ""

            if simulate:
                if not eligible_for_speculation:
                    speculated = False
                    skip_reason = "not_eligible"
                elif self.spec_policy == "scheduler" and self.scheduler is not None:
                    speculated, skip_reason, expected_benefit = self.scheduler.should_speculate(action_type)
                else:
                    speculated = True
                    skip_reason = None

                if speculated:
                    try:
                        sim_thought, sim_actions = self.generate_thought_actions(
                            i,
                            sim_running_prompt,
                            n_calls_badcalls,
                            num_actions=constants.guess_num_actions,
                            max_retries=constants.max_guess_retries,
                        )
                    except ValueError as e:
                        self.log(f"[WARN] Sim action generation failed: {e}", save_log=False)
                        speculated = False
                        skip_reason = "sim_generation_failed"
                        sim_thought = ""
                        sim_actions = []

            obs, r, done, info, normal_traj_time_taken = self.step(self.env, self.action_lowercase(action))
            obs = obs.replace("\\n", "")
            self.env.update_traj_dict_records(thought, action, obs, normal_traj_time_taken, False)
            next_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(i=i, thought=thought, action=action, obs=obs)
            running_prompt += next_step_string

            if to_print:
                self.log(f"  [Normal]  Thought: {thought}")
                self.log(f"            Action:  {action}")
                self.log(f"            Obs:     {obs[:200]}{'...' if len(obs) > 200 else ''}")
                self.log(f"            Time:    {normal_traj_time_taken:.3f}s")

            if simulate and speculated:
                sim_obs, sim_r, sim_done, sim_info, sim_traj_time_taken = self.step(
                    self.env, self.action_lowercase(action), simulate=True
                )
                sim_obs = sim_obs.replace("\n", "")
                speculator_latency = sim_traj_time_taken
                hit = bool(Metrics.compare_actions(action, sim_actions, sparse=False))
                next_sim_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(
                    i=i, thought=thought, action=action, obs=sim_obs
                )
                sim_running_prompt += next_sim_step_string

                if to_print:
                    self.log(f"  [Sim]     Thought: {sim_thought}")
                    self.log(f"            Actions: {sim_actions}")
                    self.log(f"            Obs:     {sim_obs[:200]}{'...' if len(sim_obs) > 200 else ''}")
                    self.log(f"            Time:    {sim_traj_time_taken:.3f}s")
            elif simulate:
                # Keep speculative prompt aligned with authoritative trajectory when skipped.
                sim_running_prompt = running_prompt

            # Keep sim trajectory length aligned with normal trajectory for legacy metrics.
            if simulate:
                self.env.update_traj_dict_records(
                    sim_thought,
                    sim_actions,
                    sim_obs,
                    speculator_latency,
                    True,
                )

            if self.spec_policy == "scheduler" and self.scheduler is not None:
                self.scheduler.record_step(
                    action_type=action_type,
                    tool_latency=normal_traj_time_taken,
                    speculated=speculated,
                    hit=hit,
                    speculator_latency=speculator_latency,
                )

            step_records.append(
                {
                    "step_idx": i,
                    "action_type": action_type,
                    "eligible_for_speculation": eligible_for_speculation,
                    "speculated": speculated,
                    "skip_reason": skip_reason,
                    "hit": hit,
                    "tool_latency": normal_traj_time_taken,
                    "speculator_latency": speculator_latency,
                    "expected_benefit": expected_benefit,
                }
            )

            if done:
                break

        if not done:
            obs, r, done, info, time_taken = self.step(self.env, "finish[]")
            _ = time_taken

        if to_print:
            self.log(f"\n  Result: em={info.get('em', 'N/A')}  f1={info.get('f1', 'N/A')}")

        info.update(
            {
                "n_calls": n_calls_badcalls[0],
                "n_badcalls": n_calls_badcalls[1],
                "traj": running_prompt,
                "step_records": step_records,
            }
        )
        return info

    def run(self, webthink_simulate=None, skip_done=False, idxs_override=None, seed=None):
        from google.genai.errors import ClientError, ServerError

        if self.spec_policy == "never":
            policy_simulate = False
        elif self.spec_policy in {"always", "scheduler"}:
            policy_simulate = True
        else:
            raise ValueError(f"Unknown spec_policy: {self.spec_policy}")

        if webthink_simulate is not None:
            policy_simulate = webthink_simulate

        idxs = list(range(constants.num))
        run_seed = constants.random_seed if seed is None else seed
        random.Random(run_seed).shuffle(idxs)
        idxs_to_run = idxs_override if idxs_override is not None else idxs[: constants.n_samples_to_run]

        webthink_examples = Utils.read_json(join(constants.prompts_folder, constants.prompt_file)).get(
            "webthink_simple6"
        )
        webthink_prompt = Utils.join_prompt(
            PromptTemplates.REACT_INSTRUCTION, webthink_examples, PromptTemplates.PROMPT_INSTRUCTION
        )

        rewards = []
        infos = []
        old_time = time.time()

        for i in idxs_to_run:
            self.current_index = i
            current_dir_path = join(self.base_traj_path, str(self.current_index))

            log_path = join(current_dir_path, "log.txt")
            if skip_done and os.path.exists(log_path):
                self.log(f"[SKIP] Index {i} already done", save_log=False)
                continue

            Utils.delete_file(log_path)

            try:
                info = self.webthink(
                    i,
                    prompt=webthink_prompt,
                    to_print=True,
                    n=constants.n_steps_to_run,
                    simulate=policy_simulate,
                )
            except ClientError as e:
                self.log(
                    f"[ERROR] Client error, sleeping {constants.client_error_sleep_time}s: {e}",
                    save_log=False,
                )
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(constants.client_error_sleep_time)
                continue
            except ServerError as e:
                self.log(
                    f"[ERROR] Server error, sleeping {constants.server_error_sleep_time}s: {e}",
                    save_log=False,
                )
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(constants.server_error_sleep_time)
                continue
            except ZeroDivisionError:
                self.log("[ERROR] ZeroDivisionError, skipping sample", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue
            except Exception as e:
                Utils.delete_dir(current_dir_path, nested=True)
                raise e

            rewards.append(info["em"])
            infos.append(info)

            avg_reward = sum(rewards) / len(rewards)
            avg_time = (time.time() - old_time) / len(rewards)
            self.log(
                f"\n  [Progress] {len(rewards)} samples done | "
                f"Total reward: {sum(rewards)} | "
                f"Avg reward: {avg_reward:.4f} | "
                f"Avg time/sample: {avg_time:.1f}s"
            )

            normal_observations_dict = self.env.normal_trajectory_dict
            sim_observations_dict = self.env.sim_trajectory_dict
            Utils.save_json(normal_observations_dict, join(current_dir_path, "normalobs.json"))
            Utils.save_json(info.get("step_records", []), join(current_dir_path, "step_records.json"))

            if policy_simulate:
                Utils.save_json(sim_observations_dict, join(current_dir_path, "simobs.json"))
                try:
                    metric_dict = Metrics.get_action_metrics(
                        normal_observations_dict, sim_observations_dict, sparse=False
                    )
                except ZeroDivisionError:
                    self.log("[WARN] ZeroDivisionError in metrics, skipping this sample", save_log=False)
                    Utils.delete_dir(current_dir_path, nested=True)
                    continue
                step_records = info.get("step_records")
                if step_records is not None:
                    metric_dict.update(Metrics.get_speculation_metrics_from_step_records(step_records))
            else:
                Utils.save_json(
                    {
                        "prompt": normal_observations_dict.get("prompt", ""),
                        "observations": [],
                        "thoughts": [],
                        "actions": [],
                        "time_taken": [],
                    },
                    join(current_dir_path, "simobs.json"),
                )
                metric_dict = {
                    "policy": "never",
                    "em": info.get("em"),
                    "f1": info.get("f1"),
                }

            self.log(f"  [Metrics]  idx={self.current_index}  {metric_dict}", save_log=False)
            Utils.save_json(metric_dict, join(current_dir_path, "metrics.json"))

        self.env.write()

