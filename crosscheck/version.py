import subprocess

try:
    __version__ = subprocess.check_output(["git", "describe", "--dirty", "--always"]).decode().strip()
except (ValueError, subprocess.SubprocessError):
    __version__ = "unknown"
