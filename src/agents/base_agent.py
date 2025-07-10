from abc import ABC, abstractmethod
class BaseAgent(ABC):
    def __init__(self, **kwargs):
        self.agent_id = kwargs.get("agent_id")
    @abstractmethod
    def decide(self, environment, task: object):
        pass