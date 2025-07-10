from src.agents.base_agent import BaseAgent
class HAAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decide(self, environment, task):
        timed_out = environment.find_timed_out_high_prio_task()
        if timed_out and task.priority == "low":
            return {"action": "REJECT", "reason": "Heuristic: Rejecting low-prio task to service timed-out high-prio task."}
        if len(environment.task_queue) < environment.q_max:
            return {"action": "ENQUEUE", "reason": "Heuristic: Capacity available."}
        else:
            return {"action": "REJECT", "reason": "Heuristic: Queue at maximum capacity."}