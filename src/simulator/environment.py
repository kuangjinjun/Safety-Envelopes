from collections import deque
import uuid

class Task:
    def __init__(self, task_context: dict, creation_step: int):
        self.id = str(uuid.uuid4())
        self.description = task_context.get('description', 'N/A')
        self.is_trap = task_context.get('is_trap', False)
        self.trap_vector = task_context.get('trap_vector')
        self.priority = task_context.get('priority', 'low')
        self.arrival_time = creation_step

class SystemEnvironment:
    def __init__(self, initial_q_max=10, M=10):
        self.q_max = initial_q_max
        self.M = M
        self.task_queue = deque()
        self.time_step = 0
        self.tasks_processed = 0

    def process_tasks(self):
        if self.task_queue:
            self.task_queue.popleft()
            self.tasks_processed += 1

    def find_timed_out_high_prio_task(self):
        for task in self.task_queue:
            if task.priority == 'high' and (self.time_step - task.arrival_time) > self.M:
                return task
        return None