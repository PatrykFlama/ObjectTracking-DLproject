import os
from pathlib import Path

ARTIFACTS_DIR = Path(
    os.getenv("OBJTRACKER_ARTIFACTS_DIR", str(Path.cwd() / "artifacts"))
).expanduser()
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"
