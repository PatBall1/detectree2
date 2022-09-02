"""Module contains all project wide constants."""
import logging
import os
from pathlib import Path

import dotenv

# ---------------- PATH CONSTANTS -------------------
#  Source folder path
constants_path = Path(__file__)
SRC_PATH = constants_path.parent
PROJECT_PATH = SRC_PATH.parent
dotenv.load_dotenv()

# Log relatedd paths
LOG_PATH = PROJECT_PATH / "logs"
LOG_PATH.mkdir(parents=True, exist_ok=True)

#  Data related paths
USER_PATH = Path("/gws/nopw/j04/forecol/jgcb3")
DATA_PATH = Path("/gws/nopw/j04/forecol/data")
SCRATCH_PATH = Path("/work/scratch-nopw/patball")

# ---------------- API KEYS -------------------------
# PLANET_API_KEY = os.getenv("PLANET_API_KEY")

# ---------------- LOGGING CONSTANTS ----------------
DEFAULT_FORMATTER = logging.Formatter(("%(asctime)s %(levelname)s: %(message)s "
                                       "[in %(funcName)s at %(pathname)s:%(lineno)d]"))
DEFAULT_LOG_FILE = LOG_PATH / "default_log.log"
DEFAULT_LOG_LEVEL = logging.DEBUG  # verbose logging per default

# ---------------- PROJECT CONSTANTS ----------------
# Coordinate reference systems (crs)
WGS84 = "EPSG:4326"  # WGS84 standard crs (latitude, longitude)

# ---------------- DATABASE CONSTANTS ----------------
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_CONFIG = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
