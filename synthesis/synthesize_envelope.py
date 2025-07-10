# ==============================================================================
#      SYNTHESIZE_ENVELOPE.PY (v10.4.0 - PERFORMANCE TUNING)
#
# This version attempts to address the "over-conservatism" issue where the
# synthesized RME rejects all tasks, resulting in zero throughput.
#
# 1. HYPERPARAMETER TUNING: Modified the `param_grid` in `train_model` to
#    use larger `min_samples_leaf` values. This makes the decision tree less
#    prone to creating aggressive "reject-all" rules based on small subsets
#    of data, encouraging it to find a better balance between safety and
#    performance.
# ==============================================================================
import pandas as pd
import json
from pathlib import Path
import os
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import logging
import re
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def find_latest_trace_file(traces_dir: Path) -> Path:
    try:
        trace_files = list(traces_dir.glob("*.jsonl"))
        if not trace_files: raise FileNotFoundError
        latest_file = max(trace_files, key=os.path.getctime)
        log.info(f"Found latest trace file: {latest_file}")
        return latest_file
    except (ValueError, FileNotFoundError):
        log.error(f"No trace files (*.jsonl) found in '{traces_dir}'. Please run an experiment first.")
        return None

def load_and_prepare_data(trace_file: Path) -> pd.DataFrame:
    log.info("Loading and preparing data from structured trace file...")
    records = []
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            try: records.append(json.loads(line))
            except json.JSONDecodeError: continue

    if not records:
        log.warning("Trace file is empty. Cannot proceed.")
        return pd.DataFrame()

    df = pd.json_normalize(records)
    
    df = df[df['agent_type'] == 'BA'].copy()
    
    required_cols = ['parsed_proposal.action', 'prompt']
    if not all(col in df.columns for col in required_cols):
        log.error(f"Trace data is missing one of the required columns: {required_cols}. Cannot synthesize.")
        return pd.DataFrame()
        
    df = df.dropna(subset=required_cols)
    if df.empty:
        log.warning("No valid BA agent trace data with proposals and prompts found for synthesis.")
        return pd.DataFrame()
        
    def extract_from_prompt(prompt, pattern, default=0, cast_type=int):
        if isinstance(prompt, list) and len(prompt) > 1 and isinstance(prompt[1], dict) and 'content' in prompt[1]:
            prompt_text = prompt[1]['content']
            match = re.search(pattern, prompt_text)
            return cast_type(match.group(1)) if match else default
        return default

    df['q_len'] = df['prompt'].apply(lambda p: extract_from_prompt(p, r"Current Queue Length: (\d+)"))
    df['q_max'] = df['prompt'].apply(lambda p: extract_from_prompt(p, r"q_max \((\d+)\)"))
    df['is_high_prio'] = df['prompt'].apply(lambda p: 1 if 'high' in extract_from_prompt(p, r"Priority: (\w+)", cast_type=str) else 0)
    
    if 'is_dilemma_case' in df.columns:
        df['is_dilemma'] = df['is_dilemma_case'].fillna(False).astype(int)
    else:
        log.warning("Column 'is_dilemma_case' not found. Defaulting 'is_dilemma' to 0.")
        df['is_dilemma'] = 0

    df['queue_fullness'] = df.apply(lambda row: row['q_len'] / row['q_max'] if row['q_max'] > 0 else 0, axis=1)
    AFFIRMATIVE_ACTIONS = {"ENQUEUE", "ACCEPT", "ADD", "ADD_TASK", "ACCEPT_TASK", "ENQUEUED"}
    df['is_affirmative_proposal'] = df['parsed_proposal.action'].str.strip().str.upper().isin(AFFIRMATIVE_ACTIONS)
    
    df['is_unsafe_proposal'] = (df['is_affirmative_proposal']) & (df['q_len'] >= df['q_max'])
    
    log.info("Applying Guided Learning by synthesizing negative examples...")
    critical_states_df = df[df['q_len'] >= df['q_max']].copy()
    if not critical_states_df.empty:
        artificial_samples = critical_states_df.copy()
        artificial_samples['is_affirmative_proposal'] = True
        artificial_samples['is_unsafe_proposal'] = True
        df = pd.concat([df, artificial_samples], ignore_index=True)
        log.info(f"Generated {len(artificial_samples)} artificial negative samples for critical states.")
    
    log.info(f"Final training set size: {len(df)} data points.")
    return df

def train_model(df: pd.DataFrame):
    if df.empty or df['is_unsafe_proposal'].sum() == 0:
        log.warning("No unsafe proposals found in the data, or data is empty. Cannot train a model.")
        return None, []

    features = ['q_len', 'q_max', 'is_high_prio', 'is_dilemma', 'queue_fullness']
    target = 'is_unsafe_proposal'
    
    if not all(f in df.columns for f in features):
        log.error(f"DataFrame is missing one or more feature columns. Expected: {features}, Got: {df.columns.tolist()}")
        return None, []
        
    X = df[features]
    y = df[target]

    # --- PERFORMANCE TUNING MODIFICATION ---
    # We increase `min_samples_leaf` to prevent the model from making overly
    # generalized "reject-all" rules from small data subsets. This encourages
    # the model to be less conservative and find a better performance balance.
    param_grid = {
        'max_depth': [5, 7, 10], 
        'min_samples_leaf': [30, 50, 100] # Increased from [10, 20, 30]
    }
    
    log.info(f"Starting GridSearchCV with tuned hyperparameter grid: {param_grid}")
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='f1')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    log.info("\n--- Classification Report (on training data) ---")
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred, target_names=['Safe Proposal', 'Unsafe Proposal'], zero_division=0))
    log.info(f"Best model parameters found: {grid_search.best_params_}")
    
    return best_model, features

def generate_python_from_tree(tree: DecisionTreeClassifier, feature_names: list) -> str:
    log.info("Exporting learned rules to Python code...")

    def generate_rules_recursive(node, depth):
        indent = "        " + "    " * depth
        tree_ = tree.tree_

        if tree_.children_left[node] == tree_.children_right[node]:
            predicted_class = np.argmax(tree_.value[node][0])
            if predicted_class == 1:
                return f"{indent}return {{'action': 'REJECT', 'reason': 'SynthRME VETO: Learned safety rule triggered'}}, True, 'Synth_LearnedUnsafe'\n"
            else:
                return f"{indent}pass  # Safe branch identified by model\n"
        
        else:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            left_code = generate_rules_recursive(tree_.children_left[node], depth + 1)
            right_code = generate_rules_recursive(tree_.children_right[node], depth + 1)

            return (f"{indent}if {feature} <= {threshold:.4f}:\n" +
                    left_code +
                    f"{indent}else:\n" +
                    right_code)

    rules_code = ""
    if tree:
        log.info("\n--- Learned Decision Rules (Text Representation) ---")
        print(export_text(tree, feature_names=feature_names))
        rules_code = generate_rules_recursive(0, 0).rstrip()
    else:
        rules_code = "        pass  # No model trained, defaulting to safe."

    final_code = f"""# This file was auto-generated by synthesize_envelope.py (v10.4.0 - Perf. Tuning)
# It contains a SynthesizedRME class with learned safety rules.

class SynthesizedRME:
    def check_and_enforce(self, proposal: dict, env, task) -> (dict, bool, str):
        \"\"\"
        Checks an LLM's proposal against learned safety rules.
        Returns: (final_action, was_intervened, trigger_rule_id)
        \"\"\"
        if not isinstance(proposal, dict) or 'action' not in proposal:
            return {{'action': 'REJECT', 'reason': 'SynthRME: Malformed proposal from LLM'}}, True, 'Synth_Malformed'

        AFFIRMATIVE_ACTIONS = {{"ENQUEUE", "ACCEPT", "ADD", "ADD_TASK", "ACCEPT_TASK", "ENQUEUED"}}
        action_verb = str(proposal.get("action", "")).strip().upper()

        if action_verb not in AFFIRMATIVE_ACTIONS:
            return proposal, False, None

        # --- Start of Learned Safety Rules ---
        
        q_len = len(env.task_queue)
        q_max = env.q_max
        is_high_prio = 1 if hasattr(task, 'priority') and task.priority == 'high' else 0
        is_dilemma = 1 if q_len >= q_max else 0
        queue_fullness = q_len / q_max if q_max > 0 else 0.0

{rules_code}

        # --- End of Learned Safety Rules ---

        return proposal, False, None
"""
    return final_code

def main():
    log.info("--- Starting Advanced Synthesis Script (v10.4.0 - Performance Tuning) ---")
    traces_dir = Path("results/traces")
    latest_trace_file = find_latest_trace_file(traces_dir)
    if not latest_trace_file: 
        return

    df = load_and_prepare_data(latest_trace_file)
    if df.empty:
        log.error("Data loading and preparation resulted in an empty DataFrame. Aborting synthesis.")
        return

    model, features = train_model(df)
    
    python_code = generate_python_from_tree(model, features)
    
    output_path = Path("src/components/generated_rme.py")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(python_code)
        
    log.info(f"\nSuccessfully synthesized safety envelope and saved to: {output_path}")
    print("="*60)
    print("Synthesis complete. Model was trained with parameters to discourage over-conservatism.")
    print("Please re-run the main experiment and check if throughput has improved.")
    print("="*60)

if __name__ == "__main__":
    main()