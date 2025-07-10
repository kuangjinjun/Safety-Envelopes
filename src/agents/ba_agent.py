from src.agents.base_agent import BaseAgent
import time

class BAAgent(BaseAgent):
    def __init__(self, client, prompt_manager, trace_log_queue, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.prompt_manager = prompt_manager
        self.trace_log_queue = trace_log_queue

    def decide(self, environment, task):
        context = {"q_len": len(environment.task_queue), "q_max": environment.q_max, "M": environment.M, "task": task}
        messages = self.prompt_manager.get_messages(scene='RDS', agent_type='BA_Agent', context=context)

        llm_proposal, raw_response, error = self.client.query(messages)

        if self.trace_log_queue:
            self.trace_log_queue.put({
                "timestamp": time.time(), "agent_type": "BA", "model": self.client.model_name,
                "decision_idx": environment.time_step, "prompt": messages,
                "raw_response": raw_response, "parsed_proposal": llm_proposal, "error": error
            })

        return llm_proposal