
from typing import Any, List, Literal, Optional

from pydantic import BaseModel

from custom_video_editor import CustomVideoEditor


class CustomWebsocketData(BaseModel):
    type: Literal["PROGRESS", "PRINT", "STATUS"]
    message: str | int
    desc: Optional[str] = None


class CustomTask(BaseModel):
    id: str
    video_editor: CustomVideoEditor
    status: Literal["PENDING", "RUNNING", "COMPLETED", "CANCELED"] = "PENDING"
    messages: List[Any] = []


