import uuid
from asyncio import Event

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from starlette.websockets import WebSocket

from model import CustomWebsocketData
from task_manager import TaskManager

app = FastAPI()
task_manager = TaskManager()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/")
async def add(model: CreateModel):
    input_file = model.input_file.filename if model.input_file.filename is not None else str(uuid.uuid4()) + ".mkv"
    output_file = model.output_file if model.output_file is not None else str(uuid.uuid4()) + ".mkv"
    model.output_file = output_file
    model.input_file = input_file
    return input_file
    # task_id = await task_manager.add(input_file=input_file,output_file=output_file, only_plot=model.only_plot, modal=model.modal, custom_scene=(model.custom_scene_start, model.custom_scene_end, model.custom_scene_factor))
    # return {"message": "Added ID Successfully", "id": task_id}

@app.delete("/{task_id}")
async def add(task_id: str):
    await task_manager.delete(task_id)
    return {"message": "Deleted Successfully"}

@app.websocket("/ws/{task_id}")
async def websocket_route(ws: WebSocket, task_id: str):
    await ws.accept()
    event = Event()
    custom_data = task_manager.all_tasks[task_id]
    await ws.send_json(custom_data.model_dump())
    if custom_data.status == "COMPLETED" or custom_data.status == "CANCELED":
        event.set()

    async def progress_callback(data: CustomWebsocketData):
        await ws.send_json(data.model_dump())
        if data.type == "STATUS" and (data.message == "COMPLETED" or data.message == "CANCELED"):
            event.set()

    task_manager.custom_event.subscribe(str(task_id), progress_callback)
    await event.wait()
    await ws.close()


class CreateModel(BaseModel):
    input_file: UploadFile
    output_file: str
    modal: bool
    only_plot: bool
    custom_scene_start: str
    custom_scene_end: str
    custom_scene_factor: float
