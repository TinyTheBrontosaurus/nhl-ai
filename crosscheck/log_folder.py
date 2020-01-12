import pathlib
import datetime

class LogFolder:
    folder = pathlib.Path.cwd()
    friendly_time = None
    friendly_name = None

    @classmethod
    def set_path(cls, root: pathlib.Path, friendly_name: str):
        # Setup target log folder
        cls.friendly_time = str(datetime.datetime.now()).replace(':', "-").replace(" ", "_")
        cls.friendly_name = friendly_name
        cls.folder = pathlib.Path(root) / cls.friendly_name / cls.friendly_time
