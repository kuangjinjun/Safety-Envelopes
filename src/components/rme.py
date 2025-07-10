class HandCraftedRME:
    def check_and_enforce(self, proposal: dict, env, task) -> (dict, bool, str):
        if not isinstance(proposal, dict) or 'action' not in proposal:
             return {"action": "REJECT", "reason": "RME Override: Malformed proposal from LLM."}, True, "MalformedProposal"

        AFFIRMATIVE_ACTIONS = {"ENQUEUE", "ACCEPT", "ADD", "ADD_TASK", "ACCEPT_TASK", "ENQUEUED"}
        action_verb = str(proposal.get("action", "")).strip().upper()

        if action_verb in AFFIRMATIVE_ACTIONS:
            if len(env.task_queue) >= env.q_max:
                reason = f"RME VETO (Φ1): Queue capacity violation. q_len ({len(env.task_queue)}) >= q_max ({env.q_max})."
                return {"action": "REJECT", "reason": reason}, True, "E1_Capacity"

            timed_out_high_prio_task = env.find_timed_out_high_prio_task()
            if timed_out_high_prio_task and task.priority == "low":
                reason = f"RME VETO (Φ2): Priority violation. Low-priority task proposed while high-priority task {timed_out_high_prio_task.id} has timed out."
                return {"action": "REJECT", "reason": reason}, True, "E2_Priority"

        return proposal, False, None