from src.agents.base_agent import BaseAgent
from src.components.rme import HandCraftedRME
import time
import logging

log = logging.getLogger(__name__)

try:
    from src.components.generated_rme import SynthesizedRME
    SYNTH_RME_AVAILABLE = True
    log.info("Successfully imported SynthesizedRME.")
except (ImportError, SyntaxError) as e:
    SynthesizedRME = None
    SYNTH_RME_AVAILABLE = False
    log.warning(f"SynthesizedRME not found or failed to import, OMA_Synth will be unavailable. Reason: {e}")

class OMAAgent(BaseAgent):
    def __init__(self, client, prompt_manager, trace_log_queue, use_synthesized_rme=False, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.prompt_manager = prompt_manager
        self.feedback_history = []
        self.trace_log_queue = trace_log_queue

        if use_synthesized_rme:
            if not SYNTH_RME_AVAILABLE:
                raise ImportError("Attempted to initialize OMA_Synth, but SynthesizedRME could not be imported.")
            self.rme = SynthesizedRME()
            self.agent_type_for_log = "OMA_Synth"
        else:
            self.rme = HandCraftedRME()
            self.agent_type_for_log = "OMA"

    def decide(self, environment, task):
        context = {
            "q_len": len(environment.task_queue), "q_max": environment.q_max, 
            "M": environment.M, "task": task, "rme_feedback_history": self.feedback_history
        }
        messages = self.prompt_manager.get_messages(scene='RDS', agent_type=self.agent_type_for_log, context=context)

        llm_proposal, raw_response, error = self.client.query(messages)

        if self.trace_log_queue:
            self.trace_log_queue.put({
                "timestamp": time.time(), "agent_type": self.agent_type_for_log, "model": self.client.model_name,
                "decision_idx": environment.time_step, "prompt": messages,
                "raw_response": raw_response, "parsed_proposal": llm_proposal, "error": error
            })

        if llm_proposal is None:
            safe_fallback_action = {"action": "REJECT", "reason": "API Failure Fallback"}
            return safe_fallback_action, True, llm_proposal, "APIFailure"

        final_action, intervened, triggered_rule = self.rme.check_and_enforce(llm_proposal, environment, task)

        if intervened:
            feedback = {
                "step": environment.time_step, "proposal": llm_proposal,
                "reason": final_action.get("reason"), "task": task,
                "state": {"q_len": len(environment.task_queue), "q_max": environment.q_max}
            }
            self.feedback_history.append(feedback)
            if len(self.feedback_history) > 5: self.feedback_history.pop(0)

        return final_action, intervened, llm_proposal, triggered_rule