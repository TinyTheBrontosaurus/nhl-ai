import pathlib
import datetime
import natsort
from typing import List


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

    @classmethod
    def _get_date_folders(cls, log_folder: pathlib.Path) -> List[str]:
        return natsort.natsorted(list(log_folder.iterdir()))

    @classmethod
    def get_recent_date_folder(cls, log_folder: pathlib.Path) -> pathlib.Path:
        date_folders = cls._get_date_folders(log_folder)
        return log_folder / date_folders[-1]

    @classmethod
    def get_datetime_folders(cls, log_folder: pathlib.Path) -> List[pathlib.Path]:
        recent_date_folder = cls.get_recent_date_folder(log_folder)
        return natsort.natsorted(list(recent_date_folder.iterdir()))

    @classmethod
    def get_recent_datetime_folder(cls, log_folder: pathlib.Path) -> pathlib.Path:
        datetime_folders = cls.get_datetime_folders(log_folder)
        return datetime_folders[-1]

    @classmethod
    def get_latest_log_folder(cls, log_folder: pathlib.Path) -> pathlib.Path:
        return cls.get_recent_datetime_folder(log_folder)
