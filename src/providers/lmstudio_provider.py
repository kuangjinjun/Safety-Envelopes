import httpx
import json
import time
import re
import logging
from .base_provider import BaseProvider
from typing import Optional, Tuple

log = logging.getLogger(__name__)

class LMStudioProvider(BaseProvider):
    def __init__(self, provider_config):
        api_config = provider_config.get('api_config', {})
        self.model_name = api_config.get('model_name_override', 'local_lmstudio_model')

        self.client = httpx.Client(
            base_url=api_config.get('base_url', "http://127.0.0.1:1234/v1"),
            headers={"content-type": "application/json"},
            timeout=120.0,
        )
        self.api_config = api_config

    def _extract_json_from_string(self, text: str) -> dict:
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text_to_decode = match.group(1)
        else:
            match = re.search(r'(\{[\s\S]*\})', text)
            if not match:
                raise json.JSONDecodeError("No JSON object found in response", text, 0)
            text_to_decode = match.group(0)
        return json.loads(text_to_decode)

    def query(self, messages: list) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        payload = {
            "model": "local-model", "messages": messages,
            "temperature": self.api_config.get('temperature', 0.5),
            "max_tokens": self.api_config.get('max_tokens', 512),
        }
        for attempt in range(3):
            try:
                response = self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                raw_content = response.json()['choices'][0]['message']['content']
                try:
                    parsed_json = self._extract_json_from_string(raw_content)
                    return parsed_json, raw_content, None
                except json.JSONDecodeError as e:
                    error_msg = f"JSONDecodeError: {e.msg} in response: {raw_content}"
                    log.warning(error_msg)
                    return None, raw_content, error_msg
            except httpx.RequestError as e:
                error_msg = f"Connection Error to {e.request.url}: {e}"
                log.error(f"API Error (Model: {self.model_name}, Attempt {attempt+1}/3): {error_msg}")
                if attempt < 2: time.sleep(2 ** (attempt + 2))
                else:
                    final_error = f"Could not connect to LM Studio at {self.api_config.get('base_url')}. Is the server running?"
                    log.critical(final_error)
                    return None, None, final_error
            except Exception as e:
                error_msg = f"General API Error: {str(e)}"
                log.warning(f"API Error (Model: {self.model_name}, Attempt {attempt+1}/3): {error_msg}")
                if attempt < 2: time.sleep(2 ** attempt)

        final_error = f"API request ultimately failed after 3 attempts for model {self.model_name}."
        log.error(final_error)
        return None, None, final_error