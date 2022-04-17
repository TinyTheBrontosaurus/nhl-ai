import pathlib

ROOT_FOLDER = pathlib.Path(__file__).parents[1]

SAVE_STATE_FOLDER = ROOT_FOLDER / 'rom-layout'

LOG_ROOT = ROOT_FOLDER / "log"

NEW_SAVE_STATE_FOLDER = ROOT_FOLDER / "save-states-pending"

DB_FILE = ROOT_FOLDER / "crosscheck.db"

MODEL_ROOT = ROOT_FOLDER / "models"
