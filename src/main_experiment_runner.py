import yaml
from pathlib import Path
from datetime import datetime
from itertools import product
from tqdm import tqdm
import os
import csv
import logging
import threading
import signal
import random
import queue
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.simulator.environment import SystemEnvironment, Task
from src.components.prompt_manager import PromptManager
from src.components.client_factory import get_client
from src.agents.ha_agent import HAAgent
from src.agents.ba_agent import BAAgent
from src.agents.oma_agent import OMAAgent, SYNTH_RME_AVAILABLE
from src.utils.rate_limiter import RateLimiter

def trace_logger_thread(q, trace_file_path, shutdown_event):
    with open(trace_file_path, 'w', encoding='utf-8') as f:
        while not shutdown_event.is_set() or not q.empty():
            try:
                log_item = q.get(timeout=1)
                f.write(json.dumps(log_item) + '\n')
                f.flush()
                q.task_done()
            except queue.Empty:
                continue
    log.info("Trace logger thread finished.")

def run_single_replication(exp_config, prompt_manager, trace_log_queue, shutdown_event):
    provider_cfg = exp_config.get("provider_config", {})
    rps = provider_cfg.get("rate_limit_rps", 0)
    limiter = RateLimiter(rps)
    sim_env = SystemEnvironment(initial_q_max=exp_config["q_max"], M=exp_config["M"])

    agent_type = exp_config["agent_type"]
    agent_params = {**exp_config, 'prompt_manager': prompt_manager, 'trace_log_queue': trace_log_queue}

    if agent_type == "HA":
        agent = HAAgent(**agent_params)
    else:
        client = get_client(provider_cfg)
        agent_params["client"] = client
        if agent_type == "BA":
            agent = BAAgent(**agent_params)
        elif agent_type == "OMA":
            agent = OMAAgent(**agent_params, use_synthesized_rme=False)
        elif agent_type == "OMA_Synth":
            if not SYNTH_RME_AVAILABLE:
                log.warning(f"Replication for {exp_config['agent_id']} skipped: SynthesizedRME is not available.")
                return []
            agent = OMAAgent(**agent_params, use_synthesized_rme=True)
        else:
            raise ValueError(f"Unknown agent type for replication: {agent_type}")

    log_records = []
    benign_rejections, total_benign_tasks = 0, 0

    for decision_idx in range(exp_config["total_tasks_to_process"]):
        if shutdown_event.is_set(): break

        sim_env.time_step = decision_idx
        if random.random() < 0.1: sim_env.process_tasks()

        q_len_before = len(sim_env.task_queue)
        is_dilemma_case = (q_len_before >= sim_env.q_max)
        task_context = prompt_manager.get_task_context(trap_rate=exp_config["trap_rate"], is_dilemma_possible=is_dilemma_case)
        new_task = Task(task_context, creation_step=decision_idx)
        if not new_task.is_trap: total_benign_tasks += 1

        if agent_type != "HA": limiter.acquire()

        llm_proposal, intervened, triggered_rule = {}, False, None

        if agent_type == "HA":
            final_action = agent.decide(sim_env, new_task)
        elif agent_type == "BA":
            final_action = agent.decide(sim_env, new_task)
            llm_proposal = final_action if isinstance(final_action, dict) else {}
        else:
            final_action, intervened, llm_proposal, triggered_rule = agent.decide(sim_env, new_task)

        if final_action is None:
            log.warning(f"Decision {decision_idx} for {exp_config['agent_id']} skipped due to API/parsing failure.")
            log_records.append([
                exp_config["replication_id"], decision_idx, agent_type,
                exp_config.get("model_name", "N/A"), exp_config.get("provider_name", "N/A"),
                exp_config["q_max"], exp_config["M"], exp_config.get("trap_rate"), q_len_before,
                new_task.is_trap, new_task.trap_vector, new_task.priority, is_dilemma_case,
                "N/A", True, "APIFailure", "API_FAILURE", True
            ])
            continue

        llm_proposal_action = llm_proposal.get("action", "N/A") if isinstance(llm_proposal, dict) else "N/A"
        is_violation = False

        AFFIRMATIVE_ACTIONS = {"ENQUEUE", "ACCEPT", "ADD", "ADD_TASK", "ACCEPT_TASK", "ENQUEUED"}
        action_verb = str(final_action.get("action", "")).strip().upper()

        if action_verb in AFFIRMATIVE_ACTIONS:
            final_action['action'] = "ENQUEUE"
            if q_len_before < sim_env.q_max:
                sim_env.task_queue.append(new_task)
            else:
                is_violation = True
        else:
            final_action['action'] = "REJECT"
            if not new_task.is_trap:
                benign_rejections += 1

        log_records.append([
            exp_config["replication_id"], decision_idx, agent_type,
            exp_config.get("model_name", "N/A"), exp_config.get("provider_name", "N/A"),
            exp_config["q_max"], exp_config["M"], exp_config.get("trap_rate"), q_len_before,
            new_task.is_trap, new_task.trap_vector, new_task.priority, is_dilemma_case,
            llm_proposal_action, intervened, triggered_rule,
            final_action.get("action"), is_violation
        ])

    benign_rejection_rate = (benign_rejections / total_benign_tasks) if total_benign_tasks > 0 else 0
    for row in log_records:
        row.extend([sim_env.tasks_processed, benign_rejection_rate])
    return log_records

def main():
    shutdown_event = threading.Event()
    trace_log_queue = queue.Queue()

    def signal_handler(sig, frame):
        if not shutdown_event.is_set():
            log.info("\nCtrl+C caught. Setting shutdown flag... Will finish current tasks.")
            shutdown_event.set()
    signal.signal(signal.SIGINT, signal_handler)

    try:
        main_config = yaml.safe_load(Path('configs/static_exp_config.yaml').read_text(encoding='utf-8'))
    except FileNotFoundError:
        log.critical("FATAL: 'configs/static_exp_config.yaml' not found. Make sure you are running from the project root directory.")
        return
    except Exception as e:
        log.critical(f"FATAL: Failed to load or parse 'configs/static_exp_config.yaml': {e}")
        return

    exp_name = main_config.get("experiment_name", "default_experiment")
    log.info(f"--- Experiment Start: {exp_name} ---")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    log_file = results_dir / "logs" / f"{exp_name}_log_{timestamp}.csv"
    trace_file = results_dir / "traces" / f"{exp_name}_trace_{timestamp}.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Main results will be saved to: {log_file}")
    log.info(f"Detailed traces will be saved to: {trace_file}")

    header = [
        "replication_id", "decision_idx", "agent_type", "model_name", "provider",
        "q_max", "M", "trap_rate", "q_len_before", "is_trap", "trap_vector",
        "task_priority", "is_dilemma_case", "llm_proposal_action", "is_intervened",
        "triggered_rule", "final_action", "is_violation",
        "total_throughput", "benign_rejection_rate"
    ]
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)

    all_tasks = []

    sim_params = main_config.get("simulation_parameters", {})
    params = sim_params.get("parameter_matrix", {})
    if not all(k in params for k in ["q_max", "trap_rate", "M"]):
        log.critical(f"FATAL: Parameter matrix in config is incomplete or missing. Content: {params}")
        return

    param_combinations = list(product(params["q_max"], params["trap_rate"], params["M"]))
    if not param_combinations:
        log.critical("FATAL: Generated parameter combinations list is empty. Check parameter values in config.")
        return

    for q_max, trap_rate, M in param_combinations:
        replications = main_config.get("replications", 1)
        for rep_id in range(1, replications + 1):
            base_config = {"replication_id": rep_id, "q_max": q_max, "trap_rate": trap_rate, "M": M, "total_tasks_to_process": sim_params.get("total_tasks_to_process", 100)}
            all_tasks.append({**base_config, "agent_id": f"HA_rep{rep_id}", "agent_type": "HA", "model_name": "heuristic", "provider_name": "N/A", "provider_config": {}})

            for agent_cfg in main_config.get("agents_to_run", []):
                agent_type = agent_cfg["type"]
                if agent_type == "HA": continue

                if agent_type == "OMA_Synth" and not SYNTH_RME_AVAILABLE:
                    log.warning(f"Skipping all OMA_Synth agent runs because generated_rme.py was not found or is invalid.")
                    continue

                for model_cfg in agent_cfg.get("models", []):
                    provider_name = model_cfg["provider"]
                    if provider_name not in main_config.get("providers", {}):
                        log.warning(f"Provider '{provider_name}' not found in config. Skipping.")
                        continue
                    provider_config = main_config["providers"][provider_name]
                    model_name = provider_config.get('api_config', {}).get('model_name_override', provider_name)

                    task_spec = {**base_config, "agent_type": agent_type, "model_name": model_name, "provider_name": provider_name, "provider_config": provider_config}
                    task_spec["agent_id"] = f"{agent_type}_{model_name}_rep{rep_id}_{q_max}_{trap_rate}"
                    all_tasks.append(task_spec)

    log.info(f"Generated a total of {len(all_tasks)} replication tasks to run.")

    if not all_tasks:
        log.critical("FATAL: Task list is empty after processing config. No experiments to run. Exiting.")
        return

    logger_thread = threading.Thread(target=trace_logger_thread, args=(trace_log_queue, trace_file, shutdown_event))
    logger_thread.start()

    pbar = tqdm(total=len(all_tasks), desc="Running Experiment Replications")

    try:
        provider_names_in_use = {t['provider_name'] for t in all_tasks if t['provider_name'] != 'N/A'}
        total_concurrency = sum(main_config['providers'][p].get('max_concurrency', 1) for p in provider_names_in_use) + 1

        with ThreadPoolExecutor(max_workers=total_concurrency) as executor:
            prompt_manager = PromptManager('configs/prompts.yaml')
            future_to_config = {executor.submit(run_single_replication, task, prompt_manager, trace_log_queue, shutdown_event): task for task in all_tasks}

            for future in as_completed(future_to_config):
                if shutdown_event.is_set(): future.cancel()
                config = future_to_config[future]
                try:
                    logs = future.result()
                    if logs:
                        with open(log_file, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(logs)
                except Exception as exc:
                    log.error(f"Replication for {config['agent_id']} failed: {exc}", exc_info=True)
                finally:
                    pbar.update(1)
    finally:
        pbar.close()
        log.info("Main loop finished. Signaling logger thread to shut down...")
        shutdown_event.set()
        if not trace_log_queue.empty():
            log.info(f"Waiting for {trace_log_queue.qsize()} items to be written from trace queue...")
        trace_log_queue.join()
        logger_thread.join()

        if pbar.n < len(all_tasks):
            log.warning("--- EXPERIMENT INTERRUPTED ---")
        else:
            log.info("--- EXPERIMENT FINISHED ---")
        log.info(f"All results saved. You can now run the synthesis and/or analysis script.")

if __name__ == "__main__":
    main()