from typing import Optional, Callable

from proglog import ProgressBarLogger

from task_manager import CustomWebsocketData


class MyBarLogger(ProgressBarLogger):
    def __init__(self, custom_callback: Optional[Callable]):
        super(MyBarLogger, self).__init__()
        self.custom_callback = custom_callback

    def callback(self, **changes):
        for (_, message) in changes.items():
            self.custom_callback(CustomWebsocketData(type="PRINT", message=message))

    def bars_callback(self, bar, attr, value, old_value=None):
        print(bar, attr, value, self)