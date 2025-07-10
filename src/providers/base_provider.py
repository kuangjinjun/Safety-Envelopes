from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseProvider(ABC):
    @abstractmethod
    def query(self, messages: list) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        """Returns (parsed_json, raw_response, error_message)"""
        pass