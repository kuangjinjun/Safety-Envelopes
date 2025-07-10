import yaml
import random
from jinja2 import Environment, DictLoader, select_autoescape

class PromptManager:
    def __init__(self, config_path: str = 'configs/prompts.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        try:
            macros_content = self.config['agent_prompts']['RDS']['macros']
            ba_user_template = self.config['agent_prompts']['RDS']['BA_Agent']['user_template']
            oma_user_template = self.config['agent_prompts']['RDS']['OMA_Agent']['user_template']
        except KeyError as e: raise ValueError(f"Could not find required keys in prompts.yaml. Missing: {e}")
        self.template_loader = DictLoader({'macros.jinja': macros_content, 'ba_user.jinja': ba_user_template, 'oma_user.jinja': oma_user_template})
        self.env = Environment(loader=self.template_loader, autoescape=select_autoescape())

    def get_task_context(self, trap_rate: float, is_dilemma_possible: bool) -> dict:
        if is_dilemma_possible and random.random() < trap_rate:
            task = random.choice(self.config['content_library']['trap_tasks']['dilemma'])
            return {"is_trap": True, "trap_vector": "dilemma", **task}

        if random.random() < trap_rate:
            trap_vector = random.choice(['privilege', 'state_deception', 'imminent_failure'])
            task = random.choice(self.config['content_library']['trap_tasks'][trap_vector])
            return {"is_trap": True, "trap_vector": trap_vector, **task}
        else:
            task = random.choice(self.config['content_library']['benign_tasks'])
            return {"is_trap": False, "trap_vector": None, **task}

    def get_messages(self, scene: str, agent_type: str, context: dict) -> list:
        effective_agent_type = 'OMA_Agent' if agent_type in ['OMA', 'OMA_Synth'] else 'BA_Agent'
        try:
            prompt_config = self.config['agent_prompts'][scene][effective_agent_type]
        except KeyError:
            raise ValueError(f"Prompt config for scene '{scene}' and agent '{effective_agent_type}' not found")

        messages = [{"role": "system", "content": prompt_config['system_message']}]
        template_name = 'ba_user.jinja' if effective_agent_type == 'BA_Agent' else 'oma_user.jinja'
        user_template = self.env.get_template(template_name)
        messages.append({"role": "user", "content": user_template.render(context)})
        return messages