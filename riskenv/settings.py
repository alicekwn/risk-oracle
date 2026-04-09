import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

CRYPTOCOMPARE_API = os.getenv("CRYPTOCOMPARE_API")
