import os

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
openrouter_model_name = "openai/gpt-4"
openrouter_guess_model_name = "openai/gpt-5-nano"
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# Resolve paths relative to hotpotqa/ so scripts work from any cwd.
HOTPOTQA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
prompts_folder = os.path.join(HOTPOTQA_ROOT, "prompts")
prompt_file = "prompts_naive.json"
agent_role = "Question Answering Agent"

random_seed = 248
num = 7405
n_steps_to_run = 8
n_samples_to_run = 20
client_error_sleep_time = 60
server_error_sleep_time = 60
trajectory_filenames = ["log.txt", "normalobs.json", "simobs.json", "metrics.json"]
guess_num_actions = 3
max_agent_retries = 1
max_guess_retries = 3

# Agent LLM settings
max_output_tokens = 1000
top_p = 1
temperature = 1

# Guess LLM settings
max_guess_output_tokens = 100
guess_top_p = 0.9
guess_temperature = 0.1
