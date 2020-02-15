import pathlib
import datetime
import natsort
from typing import List, Optional


class LogFolder:
    folder = pathlib.Path.cwd()
    friendly_time = None
    friendly_name = None
    start_time = datetime.datetime.now()
    latest_log_folder: pathlib.Path = None
    latest_log_folder_checked = False

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
    def get_recent_date_folder(cls, log_folder: pathlib.Path) -> Optional[pathlib.Path]:
        date_folders = cls._get_date_folders(log_folder)
        if not date_folders:
            raise FileNotFoundError
        return log_folder / date_folders[-1]

    @classmethod
    def get_datetime_folders(cls, log_folder: pathlib.Path) -> List[pathlib.Path]:
        recent_date_folder = cls.get_recent_date_folder(log_folder)
        if not recent_date_folder:
            raise FileNotFoundError
        return natsort.natsorted(list(recent_date_folder.iterdir()))

    @classmethod
    def get_recent_datetime_folder(cls, log_folder: pathlib.Path) -> Optional[pathlib.Path]:
        datetime_folders = cls.get_datetime_folders(log_folder)
        if not datetime_folders:
            raise FileNotFoundError
        return datetime_folders[-1]

    @classmethod
    def get_latest_log_folder(cls, log_folder: pathlib.Path) -> pathlib.Path:
        """
        Accessor for the latest log folder. Recommended to call this early in execution
        (that is, before a new log folder is created)
        """
        # Cache the latest log folder
        if not cls.latest_log_folder_checked:
            cls.latest_log_folder_checked = True
            cls.latest_log_folder = cls.get_recent_datetime_folder(log_folder)

        return cls.latest_log_folder
