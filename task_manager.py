import asyncio
import uuid
from typing import Dict, Literal, Callable, Any, Coroutine
from typing import List, Optional

from pydantic import BaseModel

from custom_video_editor import CustomVideoEditor


# TODO: Add Generics
class TaskManager:
    def __init__(self):
        self.tasks_queue: CustomQueue = CustomQueue()  # Queue to manage tasks
        self.all_tasks: Dict[str, CustomTask] = {}  # List to hold all tasks
        self.current_task: Optional[str] = None  # Track the currently running task
        self.custom_event: EventManager = EventManager()

    async def add(self, input_file: str, output_file: str, only_plot: bool, modal: bool,
                  custom_scene: tuple[str, str, float]) -> str:
        """Add a task to the queue and all_tasks."""
        task_id = str(uuid.uuid4())
        task = CustomVideoEditor(input_file=input_file, custom_scene=custom_scene, output_file=output_file,
                                 callback=lambda data: self.update_event(task_id, data), only_plot=only_plot,
                                 modal=modal)
        self.tasks_queue.enqueue(task_id)
        self.all_tasks[task_id] = CustomTask(video_editor=task, id=task_id)

        await self.update_event_status(task_id)
        print(f"Task '{task_id}' added.")

        # If there's no current task, start running the new task
        if self.current_task is None:
            asyncio.create_task(self.run())

        return task_id

    async def run(self):
        """Run the next task in the queue."""
        if self.tasks_queue.is_empty():
            return

        # Peek at the next task to run
        task_id: str = self.tasks_queue.peek()
        self.current_task = task_id

        self.all_tasks[task_id].status = "RUNNING"

        await self.update_event_status(task_id)
        print(f"Starting task '{task_id}'.")

        self.all_tasks[task_id].video_editor.run()

        # Mark task as completed
        self.all_tasks[task_id].status = "COMPLETED"

        await self.update_event_status(task_id)
        print(f"Task '{self.current_task}' completed.")

        # Remove the task from the queue
        self.tasks_queue.dequeue()
        self.current_task = None  # Clear the current task

        # If there are more tasks, start the next one
        if not self.tasks_queue.is_empty():
            asyncio.create_task(self.run())

    async def delete(self, task_id: str):
        """Delete a task from the queue if not running."""
        if self.current_task and self.current_task == task_id:
            print(f"Cannot delete task '{task_id}'. It is currently running.")
            return

        # Remove the task from the queue and update the status in all_tasks
        self.tasks_queue.delete(task_id)
        self.all_tasks[task_id].status = "CANCELED"

        await self.update_event_status(task_id)
        print(f"Task '{task_id}' cancelled.")

    async def update_event(self, task_id: str, data: CustomWebsocketData):
        await self.custom_event.notify_progress(task_id, data)
        self.all_tasks[task_id].messages.append(data)

    async def update_event_status(self, task_id: str):
        status = self.all_tasks[task_id].status
        await self.update_event(task_id, CustomWebsocketData(type="STATUS", message=status))
