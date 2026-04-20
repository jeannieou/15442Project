import os
import re
import openai
from google import genai
from google.genai import types

from . import constants


class LLMClient:
    """Unified LLM client supporting Gemini, OpenAI, OpenRouter, and DeepSeek APIs."""

    def __init__(self, model_name, temperature, max_tokens, top_p):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.provider, self.resolved_model = self._resolve_model_provider(model_name)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", getattr(constants, "openai_api_key", None))
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", getattr(constants, "openrouter_api_key", None))
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", getattr(constants, "gemini_api_key", None))
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", getattr(constants, "deepseek_api_key", None))

        # Lazily initialized clients (avoid requiring unrelated keys).
        self._openai_client = None
        self._openrouter_client = None
        self._gemini_client = None
        self._deepseek_client = None

    @staticmethod
    def _resolve_model_provider(model_name):
        """Infer provider and provider-specific model id."""
        if model_name.startswith("openai/"):
            return "openai", model_name.split("/", 1)[1]

        if model_name.startswith("gpt-"):
            return "openai", model_name

        if model_name.startswith("gemini"):
            return "gemini", model_name

        if model_name.startswith("deepseek/"):
            return "deepseek", model_name.split("/", 1)[1]

        if model_name.startswith("deepseek-"):
            return "deepseek", model_name

        if "/" in model_name:
            provider_prefix = model_name.split("/", 1)[0]
            if provider_prefix == "google":
                # e.g. google/gemini-* usually comes from OpenRouter naming.
                return "openrouter", model_name
            if provider_prefix == "deepseek":
                return "deepseek", model_name.split("/", 1)[1]
            return "openrouter", model_name

        return "openai", model_name

    def _get_openai_client(self):
        if self._openai_client is None:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI API key is missing. Set OPENAI_API_KEY "
                    "or constants.openai_api_key."
                )
            self._openai_client = openai.OpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def _get_openrouter_client(self):
        if self._openrouter_client is None:
            if not self.openrouter_api_key:
                raise ValueError(
                    "OpenRouter API key is missing. Set OPENROUTER_API_KEY "
                    "or constants.openrouter_api_key."
                )
            self._openrouter_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )
        return self._openrouter_client

    def _get_gemini_client(self):
        if self._gemini_client is None:
            if not self.gemini_api_key:
                raise ValueError(
                    "Gemini API key is missing. Set GEMINI_API_KEY "
                    "or constants.gemini_api_key."
                )
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        return self._gemini_client

    def _get_deepseek_client(self):
        if self._deepseek_client is None:
            if not self.deepseek_api_key:
                raise ValueError(
                    "DeepSeek API key is missing. Set DEEPSEEK_API_KEY "
                    "or constants.deepseek_api_key."
                )
            self._deepseek_client = openai.OpenAI(
                base_url="https://api.deepseek.com",
                api_key=self.deepseek_api_key,
            )
        return self._deepseek_client

    def call(self, prompt, stop=None):
        if self.provider == "gemini":
            return self._gemini_call(prompt, stop)
        elif self.provider == "openai":
            return self._openai_call(prompt, stop)
        elif self.provider == "deepseek":
            return self._deepseek_call(prompt, stop)
        else:
            return self._openrouter_call(prompt, stop)

    def _token_limit_kwargs(self):
        # GPT-5 chat-completions models require max_completion_tokens.
        if self.resolved_model.startswith("gpt-5"):
            return {"max_completion_tokens": self.max_tokens}
        return {"max_tokens": self.max_tokens}

    def _gemini_call(self, prompt, stop):
        if stop is not None:
            config = types.GenerateContentConfig(stop_sequences=stop)
        else:
            config = types.GenerateContentConfig()
        gemini_client = self._get_gemini_client()
        response = gemini_client.models.generate_content(
            model=self.resolved_model, contents=prompt, config=config
        )
        return str(response.text)

    def _openai_call(self, prompt, stop):
        openai_client = self._get_openai_client()
        kwargs = dict(
            model=self.resolved_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        kwargs.update(self._token_limit_kwargs())
        response = self._chat_with_unsupported_param_retry(openai_client, kwargs)
        return response.choices[0].message.content

    def _openrouter_call(self, prompt, stop):
        openrouter_client = self._get_openrouter_client()
        kwargs = dict(
            model=self.resolved_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        kwargs.update(self._token_limit_kwargs())
        response = self._chat_with_unsupported_param_retry(openrouter_client, kwargs)
        return response.choices[0].message.content

    def _deepseek_call(self, prompt, stop):
        deepseek_client = self._get_deepseek_client()
        kwargs = dict(
            model=self.resolved_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        kwargs.update(self._token_limit_kwargs())
        response = self._chat_with_unsupported_param_retry(deepseek_client, kwargs)
        return response.choices[0].message.content

    def _chat_with_unsupported_param_retry(self, client, kwargs):
        """Retry when API reports an unsupported parameter.

        This avoids blind token-parameter toggling that can mask the real error.
        """
        params = dict(kwargs)
        for _ in range(6):
            try:
                return client.chat.completions.create(**params)
            except openai.BadRequestError as e:
                msg = str(e)
                m = re.search(r"Unsupported parameter:\s*'([^']+)'", msg)
                if m:
                    bad_param = m.group(1)
                    if bad_param in params:
                        params.pop(bad_param, None)
                        # Token-limit compatibility switch when needed.
                        if bad_param == "max_tokens":
                            params["max_completion_tokens"] = self.max_tokens
                        elif bad_param == "max_completion_tokens":
                            params["max_tokens"] = self.max_tokens
                        continue
                    raise

                m_val = re.search(r"Unsupported value:\s*'([^']+)'.*param':\s*'([^']+)'", msg)
                if m_val:
                    bad_value = m_val.group(1)
                    bad_param = m_val.group(2)
                    if bad_param == "temperature":
                        params["temperature"] = 1
                        continue
                    # Generic fallback: drop unsupported-valued optional controls.
                    if bad_param in params:
                        params.pop(bad_param, None)
                        continue
                    # Sometimes API returns value string as param by itself.
                    if bad_value in params:
                        params.pop(bad_value, None)
                        continue
                raise
        raise RuntimeError("Failed to call chat.completions after unsupported-parameter retries")
