import pathlib
import datetime

class LogFolder:
    folder = pathlib.Path.cwd()
    friendly_time = None
    friendly_name = None
    start_time = datetime.datetime.now()

    @classmethod
    def set_path(cls, root: pathlib.Path, friendly_name: str):
        # Setup target log folder
        the_date = cls.start_time.date()
        cls.friendly_time = str(datetime.datetime.now()).replace(':', "-").replace(" ", "_")
        cls.friendly_name = friendly_name
        cls.folder = pathlib.Path(root) / cls.friendly_name / str(the_date) / cls.friendly_time

        # Create the folders
        cls.folder.mkdir(parents=True, exist_ok=True)
