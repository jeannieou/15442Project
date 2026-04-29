"""
Chess Speculative Game Runner

A chess game runner that implements speculative execution
where a guess model predicts opponent moves and prepare responses in parallel.
"""

import re
import time
import uuid
import os
import math
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from typing import Dict, List, Optional, Tuple, Any

import chess
from openai import OpenAI

import textarena as ta
from utils import Utils
import yaml


class Config:
    """Configuration management with YAML support"""
    
    def __init__(self, config_path: Optional[str] = "./config.yml"):
        if config_path and config_path.endswith('.yml'):
            self._load_from_yaml(config_path)
    
    def _load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # API Configuration
            openai_key = (config['api']['openai']['key'] or "").strip()
            openrouter_key = (config['api']['openrouter']['key'] or "").strip()
            self.openai_api_key = openai_key or os.getenv("OPENAI_API_KEY", "").strip()
            self.openrouter_api_key = openrouter_key or os.getenv("OPENROUTER_API_KEY", "").strip()
            
            # Model Configuration
            self.openai_model_name = config['models']['openai']['main']
            self.openai_guess_model_name = config['models']['openai']['guess']
            self.openrouter_model_name = config['models']['openrouter']['main']
            self.openrouter_guess_model_name = config['models']['openrouter']['guess']
            
            # Game Configuration
            self.num_chess_players = config['game']['num_players']
            self.client_error_sleep_time = config['game']['error_sleep_time']
            self.server_error_sleep_time = config['game']['error_sleep_time']
            self.stop_after = config['game']['stop_after']
            self.agent_name0 = config['game']['agent_name0']
            self.agent_name1 = config['game']['agent_name1']
            self.num_guesses = config['game']['num_guesses']

            # Guess Model Configuration
            self.guess_model_name = config['guess']['model_name']
            self.guess_provider = config['guess']['provider']

            # Path Configuration
            self.trajectories_path = config['paths']['trajectories']

            # Prompts Configuration
            self.standard_game_prompt = config['prompts']['standard_game']
            self.guess_prompt = config['prompts']['guess']
            self.retry_prompt = config['prompts']['retry']

            # Optional scheduler configuration (state-aware speculation gate)
            scheduler = config.get('scheduler', {})
            weights = scheduler.get('weights', {})
            self.scheduler_enabled = bool(scheduler.get('enabled', False))
            self.scheduler_threshold = float(scheduler.get('threshold', 0.25))
            self.scheduler_base_hit_rate = float(scheduler.get('base_hit_rate', 0.25))
            self.scheduler_lambda_cost = float(scheduler.get('lambda_cost', 0.2))
            self.scheduler_actor_time_base = float(scheduler.get('actor_time_base', 52.46))
            self.scheduler_actor_time_per_legal = float(scheduler.get('actor_time_per_legal', 4.83))
            self.scheduler_spec_time_base = float(scheduler.get('spec_time_base', 109.07))
            self.scheduler_spec_time_per_legal = float(scheduler.get('spec_time_per_legal', 2.81))
            self.scheduler_spec_cost_scale_per_guess = float(
                scheduler.get('spec_cost_scale_per_guess', 0.35)
            )

            self.scheduler_w_in_check = float(weights.get('in_check', 0.9))
            self.scheduler_w_log_legal = float(weights.get('log_legal', -0.4))
            self.scheduler_w_checking_ratio = float(weights.get('checking_ratio', 1.1))
            self.scheduler_w_capture_ratio = float(weights.get('capture_ratio', 0.7))
            self.scheduler_w_mate_in_1 = float(weights.get('mate_in_1', 2.0))
            self.scheduler_w_promotion = float(weights.get('promotion', 0.8))

            needs_openai_key = (
                self.agent_name0 == "OpenAI"
                or self.agent_name1 == "OpenAI"
                or self.guess_model_name.startswith("gpt")
                or self.guess_model_name.startswith("o")
            )
            needs_openrouter_key = (
                self.agent_name0 == "OpenRouter"
                or self.agent_name1 == "OpenRouter"
                or "/" in self.guess_model_name
            )

            if needs_openai_key and not self.openai_api_key:
                raise ValueError(
                    "Missing OpenAI API key. Set api.openai.key in config.yml "
                    "or export OPENAI_API_KEY."
                )
            if needs_openrouter_key and not self.openrouter_api_key:
                raise ValueError(
                    "Missing OpenRouter API key. Set api.openrouter.key in config.yml "
                    "or export OPENROUTER_API_KEY."
                )

        except Exception as e:
            raise ValueError(f"Error loading YAML config '{config_path}': {e}") from e

class ChessActionCleaner:
    """Utility class for cleaning and validating chess actions"""
    
    UCI_PATTERN = re.compile(r'\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]')
    
    @classmethod
    def clean_action(cls, action: Optional[str]) -> Optional[str]:
        """
        Clean and validate a chess action string in UCI format.
        """
        if action is None:
            return None
            
        match = cls.UCI_PATTERN.search(action)
        if match:
            return f'[{match.group(1)}]'
        
        return None
    
    @classmethod
    def clean_actions(cls, action: Optional[str]) -> List[str]:
        """
        Clean and validate multiple chess action strings in UCI format.
        """
        if action is None:
            return []
            
        matches = cls.UCI_PATTERN.findall(action)
        return [f'[{move}]' for move in matches]


class GameLogger:
    
    def __init__(self, base_path: str, run_id: str):
        self.base_path = base_path
        self.run_id = run_id
        self.log_path = join(base_path, str(run_id), "log.txt")
    
    def log(self, level: str, *args, save_log: bool = True) -> None:
        """Log message with specified level"""
        message = f"{level.upper()} {' '.join(str(arg) for arg in args)}\n"
        print(message, end='')
        
        if save_log:
            Utils.append_file(message, self.log_path)


class AgentManager:
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
    
    def create_agents(self, agent0_name: str, agent1_name: str) -> Dict[int, Any]:

        agents = {}
        
        for i, agent_name in enumerate([agent0_name, agent1_name]):
            if agent_name == "OpenRouter":
                agents[i] = ta.agents.OpenRouterAgent(
                    model_name=self.config.openrouter_model_name,
                    system_prompt=self.config.standard_game_prompt,
                    api_key=self.config.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    verbose=False
                )
            elif agent_name == "OpenAI":
                agents[i] = ta.agents.OpenAIAgent(
                    model_name=self.config.openai_model_name,
                    system_prompt=self.config.standard_game_prompt,
                    api_key=self.config.openai_api_key,
                    base_url="https://api.openai.com/v1",
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
        
        return agents
    
    def call_guess_llm(self, prompt: str, model_name: str, retries: int = 3) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        for attempt in range(retries):
            try:
                if model_name.startswith("gpt") or model_name.startswith("o"):
                    response = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.config.standard_game_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning_effort="low",
                        n=1
                    )
                    input_tokens, output_tokens, total_tokens = None, None, None
                    usage = response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                    return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens
                
                elif "/" in model_name:  # OpenRouter model
                    response = self.openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.config.standard_game_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning_effort="low"
                    )
                    input_tokens, output_tokens, total_tokens = None, None, None
                    usage = response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                    return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens
                
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    return None, None, None, None
        
        return None, None, None, None


class SpeculativeChessRunner:
    """
    Main class for running speculative chess games.
    
    This runner implements parallel speculation where a guess model predicts
    the opponent's move while the actual agent is thinking, allowing for
    faster gameplay through speculative execution.
    """
    
    def __init__(self, config: Config):
        """Initialize the chess runner with configuration (reads all settings from config)."""
        self.config = config
        self.agent_manager = AgentManager(self.config)
        self.agent0_name = config.agent_name0
        self.agent1_name = config.agent_name1
        self.guess_model_name = config.guess_model_name
        self.num_guesses = config.num_guesses

        self.current_run_id: Optional[str] = None
        base = self.config.trajectories_path
        self.base_traj_path = f"{base.rstrip('/')}/{self.agent0_name}_vs_{self.agent1_name}_guess{self.guess_model_name}"
        self.logger: Optional[GameLogger] = None
        
        # Initialize environment
        self.env = self._create_environment()
        print("Chess environment initialized successfully")
    
    def _create_environment(self) -> ta.Env:
        env = ta.make(env_id="Chess-v0")
        return env
    
    def _get_valid_moves(self) -> List[str]:
        """Get list of valid moves in UCI format"""
        return [f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1.0 - p))

    def _extract_scheduler_features(self, board: chess.Board) -> Dict[str, Any]:
        legal_moves = list(board.legal_moves)
        n_legal = len(legal_moves)
        checking_moves = 0
        capture_moves = 0
        promotion_moves = 0
        mate_in_1_moves = 0

        for move in legal_moves:
            if board.is_capture(move):
                capture_moves += 1
            if move.promotion is not None:
                promotion_moves += 1

            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_check():
                checking_moves += 1
            if temp_board.is_checkmate():
                mate_in_1_moves += 1

        return {
            "in_check": board.is_check(),
            "n_legal": n_legal,
            "checking_ratio": (checking_moves / n_legal) if n_legal else 0.0,
            "capture_ratio": (capture_moves / n_legal) if n_legal else 0.0,
            "promotion_available": promotion_moves > 0,
            "mate_in_1_available": mate_in_1_moves > 0,
        }

    def _scheduler_decision(self, board: chess.Board) -> Tuple[bool, Dict[str, Any]]:
        if not self.config.scheduler_enabled:
            return True, {"enabled": False, "should_speculate": True}

        features = self._extract_scheduler_features(board)
        n_legal = max(1, int(features["n_legal"]))

        z = (
            self._logit(self.config.scheduler_base_hit_rate)
            + self.config.scheduler_w_in_check * (1.0 if features["in_check"] else 0.0)
            + self.config.scheduler_w_log_legal * math.log(float(n_legal) / 20.0)
            + self.config.scheduler_w_checking_ratio * float(features["checking_ratio"])
            + self.config.scheduler_w_capture_ratio * float(features["capture_ratio"])
            + self.config.scheduler_w_mate_in_1 * (1.0 if features["mate_in_1_available"] else 0.0)
            + self.config.scheduler_w_promotion * (1.0 if features["promotion_available"] else 0.0)
        )

        p_single = self._sigmoid(z)
        k = max(1, int(self.num_guesses))
        p_hit_k = 1.0 - ((1.0 - p_single) ** k)

        t_actor_hat = (
            self.config.scheduler_actor_time_base
            + self.config.scheduler_actor_time_per_legal * n_legal
        )
        spec_scale = 1.0 + max(0, k - 1) * self.config.scheduler_spec_cost_scale_per_guess
        t_spec_hat = (
            self.config.scheduler_spec_time_base
            + self.config.scheduler_spec_time_per_legal * n_legal
        ) * spec_scale

        score = p_hit_k * t_actor_hat - self.config.scheduler_lambda_cost * t_spec_hat
        should_speculate = (p_hit_k >= self.config.scheduler_threshold) and (score > 0.0)

        return should_speculate, {
            "enabled": True,
            "threshold": self.config.scheduler_threshold,
            "lambda_cost": self.config.scheduler_lambda_cost,
            "p_single": p_single,
            "p_hit_k": p_hit_k,
            "t_actor_hat": t_actor_hat,
            "t_spec_hat": t_spec_hat,
            "score": score,
            "should_speculate": should_speculate,
            "num_guesses": k,
            "features": features,
        }
    

    def _guess_actions(self, observation: str, retries: int = 3) -> Optional[Tuple[List[str], float, int, int, int]]:
        start_pred_time = time.perf_counter()
        prompt = observation + self.config.guess_prompt.format(num_guesses=self.num_guesses)
        raw_output, input_tokens, output_tokens, total_tokens = self.agent_manager.call_guess_llm(prompt, self.guess_model_name, retries)
        if raw_output is None or input_tokens is None or output_tokens is None or total_tokens is None:
            return None
        end_pred_time = time.perf_counter()
        prediction_time = end_pred_time - start_pred_time
        
        if self.logger:
            self.logger.log("SIMULATION GUESS OUTPUT", raw_output)
        
        return ChessActionCleaner.clean_actions(raw_output), prediction_time, input_tokens, output_tokens, total_tokens
    
    def _agent_call_with_retry(
        self, 
        agent: Any, 
        observation: str, 
        player_id: int, 
        valid_moves: List[str],
        retries: int = 3
    ) -> Tuple[Optional[str], int, int, int]:
        role = "White" if player_id == 0 else "Black"
        
        for attempt in range(retries):
            raw_action, input_tokens, output_tokens, total_tokens = agent(observation)
            cleaned_action = ChessActionCleaner.clean_action(raw_action)
            
            if cleaned_action and cleaned_action in valid_moves:
                return cleaned_action, input_tokens, output_tokens, total_tokens
            
            if self.logger:
                self.logger.log("RETRY", f"Attempt {attempt + 1} failed for {role}")
            
            observation += self.config.retry_prompt.format(
                attempt=attempt + 1, 
                role=role
            )
        
        return None, 0, 0, 0
    
    def _current_agent_task(self, agent: Any, observation: str, player_id: int) -> Tuple[Optional[str], float, int, int, int]:
        """Execute the current agent's move selection"""
        start_time = time.perf_counter()
        valid_moves = self._get_valid_moves()

        role = "White" if player_id == 0 else "Black"

        truncated_observation = f"[GAME] You are playing as {role} in a game of Chess. Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n[GAME] The current board is:\n{Utils.board_with_coords(self.env.state.game_state['board'])}\n[GAME] The valid moves are: {valid_moves}."
        move, input_tokens, output_tokens, total_tokens = self._agent_call_with_retry(agent, truncated_observation, player_id, valid_moves)
        end_time = time.perf_counter()

        
        return move, end_time - start_time, input_tokens, output_tokens, total_tokens
    
    def _speculation_task(self, agent: Any, observation: str, player_id: int) -> Tuple[List[str], List[str], List[float], List[float], List[float], List[int], List[int], List[int], List[int], List[int], List[int], Dict[str, Any]]:
        """Execute speculation: predict opponent move and prepare response.
        The predictions are the speculated opponent moves, the speculations are the current agent's moves based on the predictions."""
        board = self.env.state.game_state["board"]
        should_speculate, scheduler_info = self._scheduler_decision(board)
        if self.logger:
            self.logger.log("SCHEDULER", Utils.dict_to_str(scheduler_info))
        if not should_speculate:
            return [], [], [], [], [], [], [], [], [], [], [], scheduler_info

        role = "White" if player_id == 0 else "Black"
        valid_moves = self._get_valid_moves()

        truncated_observation = f"[GAME] You are playing as {role} in a game of Chess. Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n[GAME] The current board is:\n{Utils.board_with_coords(self.env.state.game_state['board'])}\n[GAME] The valid moves are: {valid_moves}."

        prediction_results = self._guess_actions(truncated_observation, retries=3)

        if prediction_results is None:
            scheduler_info["guess_failed"] = True
            return [], [], [], [], [], [], [], [], [], [], [], scheduler_info

        predictions = prediction_results[0]
        individual_prediction_times = [prediction_results[1]] * len(predictions)
        input_prediction_tokens = [prediction_results[2]] * len(predictions)
        output_prediction_tokens = [prediction_results[3]] * len(predictions)
        total_prediction_tokens = [prediction_results[4]] * len(predictions)

        # Remove None predictions and get unique predictions
        valid_predictions = [p for p in predictions]
        valid_prediction_times = [individual_prediction_times[i] for i in range(len(individual_prediction_times))]

        if not valid_predictions:
            scheduler_info["empty_predictions"] = True
            return [], [], [], [], [], [], [], [], [], [], [], scheduler_info

        # Simulate the predicted moves in parallel
        with ThreadPoolExecutor(max_workers=len(valid_predictions)) as executor:
            speculation_futures = [
                executor.submit(self._simulate_and_speculate, agent, observation, player_id, prediction)
                for prediction in valid_predictions
            ]
            speculations_results = [future.result() for future in speculation_futures]

        print(f"Speculations: {speculations_results}")
        
        speculations: List[str] = []
        individual_speculation_times: List[float] = []
        input_speculation_tokens: List[int] = []
        output_speculation_tokens: List[int] = []
        total_speculation_tokens: List[int] = []

        for speculation in speculations_results:
            if speculation is not None and speculation[0] is not None:
                speculations.append(speculation[0])
                individual_speculation_times.append(speculation[1])
                input_speculation_tokens.append(speculation[2])
                output_speculation_tokens.append(speculation[3])
                total_speculation_tokens.append(speculation[4])

        total_times: List[float] = []
        for i in range(len(valid_prediction_times)):
            pred_time = individual_prediction_times[i]
            spec_time = individual_speculation_times[i]
            if pred_time is not None and spec_time is not None:
                total_times.append(pred_time + spec_time)

        return valid_predictions, speculations, valid_prediction_times, individual_speculation_times, total_times, input_prediction_tokens, output_prediction_tokens, total_prediction_tokens, input_speculation_tokens, output_speculation_tokens, total_speculation_tokens, scheduler_info
    
    def _simulate_and_speculate(
        self, 
        agent: Any, 
        observation: str, 
        player_id: int, 
        predicted_move: str
    ) -> Tuple[Optional[str], float, int, int, int]:
        """Simulate predicted move and generate speculative response"""

        start_time_speculate = time.perf_counter()
        # Execute predicted move on board
        move_uci = predicted_move.lower().replace("[", "").replace("]", "")
        predicted_chess_move = chess.Move.from_uci(move_uci)
        # Make a copy of the board and push the move to the copy
        board_copy = self.env.state.game_state["board"].copy()
        board_copy.push(predicted_chess_move)
        
        # Build new observation with board state
        board_str = Utils.board_with_coords(board_copy)     
        valid_moves = [f'[{move.uci()}]' for move in board_copy.legal_moves]
        valid_moves_in_string = ", ".join(valid_moves)
        spec_role = "White" if player_id == 1 else "Black"
        new_observation = f"[GAME] You are playing as {spec_role} in a game of Chess. Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n[GAME] The current board is:\n{board_str}\n[GAME] The valid moves are: {valid_moves_in_string}."

        # Get speculative move
        speculation, input_tokens, output_tokens, total_tokens = self._agent_call_with_retry(agent, new_observation, player_id, valid_moves)
        
        if self.logger:
            self.logger.log("SIMULATION SPECULATION OUTPUT", speculation)
        
        return speculation, time.perf_counter() - start_time_speculate, input_tokens, output_tokens, total_tokens
    
    def _execute_game_loop(
        self,
        agents: Dict[int, Any],
        stop_after: Optional[int] = None
    ) -> Tuple[Dict[int, Any], Any, Any, float, float]:
        """Main game execution loop"""
        self.env.reset(num_players=self.config.num_chess_players)
        
        steps_info: Dict[int, Dict[str, Any]] = {}
        step_count = 0
        done = False
        is_initial_step = True
        
        current_agent = agents[0]
        other_agent = agents[1]
        regular_time = 0.0
        speculative_time = 0.0
        temp_time_holder = 0.0

        # Initialize game state
        player_id, observation = self.env.get_observation()
        
        while not done:
            # Handle prediction matching from previous step
            if not is_initial_step:
                prev_predictions = steps_info[step_count - 1]["current_pred"]
                prev_speculations = steps_info[step_count - 1]["current_spec"]
                prev_move = steps_info[step_count - 1]["current_move"]

            player_id, observation = self.env.get_observation()

            # Use the first speculation that matches the previous move
            speculation_hit = False
            current_move = None
            if not is_initial_step and prev_predictions:
                for i, pred in enumerate(prev_predictions):
                    if pred == prev_move and i < len(prev_speculations):
                        current_move = prev_speculations[i]
                        speculation_hit = True
                        break

            if speculation_hit:
                current_predictions : List[str] = []
                current_speculations : List[str] = []     
                time_taken1: float = 0
                times_taken2: List[float] = []
                prediction_times: List[float] = []
                speculation_times: List[float] = []
                input_prediction_tokens: List[int] = []
                output_prediction_tokens: List[int] = []
                total_prediction_tokens: List[int] = []
                input_speculation_tokens: List[int] = []
                output_speculation_tokens: List[int] = []
                total_speculation_tokens: List[int] = []
                scheduler_info: Dict[str, Any] = {
                    "enabled": self.config.scheduler_enabled,
                    "should_speculate": False,
                    "reason": "reused_previous_speculation",
                }
                input_tokens1: int = 0
                output_tokens1: int = 0
                total_tokens1: int = 0
                regular_time += temp_time_holder
            else:
                # Run parallel execution
                with ThreadPoolExecutor() as executor:
                    current_future = executor.submit(
                        self._current_agent_task, current_agent, observation, player_id
                    )
                    speculation_future = executor.submit(
                        self._speculation_task, other_agent, observation, player_id
                    )
                    
                    current_move, time_taken1, input_tokens1, output_tokens1, total_tokens1 = current_future.result()

                    current_predictions, current_speculations, prediction_times, speculation_times, times_taken2, input_prediction_tokens, output_prediction_tokens, total_prediction_tokens, input_speculation_tokens, output_speculation_tokens, total_speculation_tokens, scheduler_info = speculation_future.result()
                    
                    if is_initial_step:
                        is_initial_step = False
                
                # Update timing counters
                regular_time += time_taken1

                if current_move in current_predictions:
                    speculation_hit_index = current_predictions.index(current_move)
                    speculative_time += max(time_taken1, times_taken2[speculation_hit_index] or 0)
                    temp_time_holder = speculation_times[speculation_hit_index] or 0
                else:
                    speculative_time += time_taken1
            
            # Record step information
            steps_info[step_count] = {
                "player_id": player_id,
                "current_observation": observation,
                "current_move": current_move,
                "current_pred": current_predictions,
                "current_spec": current_speculations,
                "time_taken_current_agent": time_taken1,
                "time_taken_other_agent": times_taken2,
                "time_taken_prediction": prediction_times,
                "time_taken_speculation": speculation_times,
                "speculation_hit": speculation_hit,
                "num_predictions": len(current_predictions) if current_predictions else 0,
                "input_tokens_current_agent": input_tokens1,
                "output_tokens_current_agent": output_tokens1,
                "total_tokens_current_agent": total_tokens1,
                "input_tokens_prediction": input_prediction_tokens,
                "output_tokens_prediction": output_prediction_tokens,
                "total_tokens_prediction": total_prediction_tokens,
                "input_tokens_speculation": input_speculation_tokens,
                "output_tokens_speculation": output_speculation_tokens,
                "total_tokens_speculation": total_speculation_tokens,
                "scheduler_info": scheduler_info,
            }
            
            if self.logger:
                self.logger.log("INFO", f"STEP {step_count}:", Utils.dict_to_str(steps_info[step_count]))
                self.logger.log('-' * 100)
            
            # Execute move
            done, info = self.env.step(current_move)
            step_count += 1
            
            if stop_after and step_count >= stop_after:
                break
            
            # Swap agents for next turn
            current_agent, other_agent = other_agent, current_agent
        
        rewards, game_info = self.env.close()
        return steps_info, rewards, game_info, regular_time, speculative_time
    
    def run(self, stop_after: int = 20) -> None:
        """Run a complete chess game with speculative execution"""
        # Setup run
        self.current_run_id = str(uuid.uuid4())
        self.logger = GameLogger(self.base_traj_path, self.current_run_id)
        
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        
        # Create agents
        agents = self.agent_manager.create_agents(self.agent0_name, self.agent1_name)
        
        self.logger.log("INFO", f"Starting run {self.current_run_id} with guess model: {self.guess_model_name}")
        
        try:
            # Execute game
            steps_info, rewards, game_info, regular_time, speculative_time = self._execute_game_loop(
                agents, stop_after
            )
            
            # Save results
            Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
            Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
            Utils.save_json(game_info, join(current_dir_path, "game_info.json"))
            Utils.save_json(regular_time, join(current_dir_path, "time_checker_regular.json"))
            Utils.save_json(speculative_time, join(current_dir_path, "time_checker_speculate.json"))
            
            self.logger.log("INFO", f"Run completed for {self.current_run_id}")
            
        except Exception as e:
            if self.logger:
                self.logger.log("ERROR", str(e))
            raise


def main():
    """Main execution function"""
    import argparse
    p = argparse.ArgumentParser(description="Run speculative chess (generate trajectories with built-in speculation).")
    p.add_argument("--config", default="config.yml", help="Path to config YAML (default: config.yml)")
    p.add_argument("--trajectories-dir", default=None, help="Output directory for trajectories (overrides config and TRAJECTORIES_DIR)")
    p.add_argument("--stop-after", type=int, default=None, help="Stop after N steps (default: from config)")
    args = p.parse_args()

    config = Config(args.config)
    if args.trajectories_dir is not None:
        config.trajectories_path = args.trajectories_dir.rstrip("/")

    runner = SpeculativeChessRunner(config=config)
    stop_after = args.stop_after if args.stop_after is not None else config.stop_after
    runner.run(stop_after=stop_after)

    print("Run completed")


if __name__ == "__main__":
    main()
