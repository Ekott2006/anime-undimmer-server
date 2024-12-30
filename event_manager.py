
from typing import Any, Callable, Coroutine, Dict, List

class EventManager:
    def __init__(self):
        self.subscribers: Dict[
            str, List[Callable[[Any], Coroutine[Any, Any, None]]]] = {}  # Dictionary to hold subscribers for the task

    def subscribe(self, task_id: str, callback: Callable[[Any], Coroutine[Any, Any, None]]):
        """Subscribe to progress updates for a task."""
        if task_id not in self.subscribers:
            self.subscribers[task_id] = []
        self.subscribers[task_id].append(callback)

    async def notify_progress(self, task_id: str, data: Any):
        """Notify all subscribers about data changes."""
        if task_id in self.subscribers:
            for callback in self.subscribers[task_id]:
                await  callback(data)

