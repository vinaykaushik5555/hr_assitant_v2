from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import
load_dotenv()

# Directory where HR policies are stored (admin upload)
POLICY_DIR = Path(os.getenv("POLICY_DIR", "data/policies")).resolve()

# Directory where Chroma vector DB is stored
POLICY_INDEX_DIR = Path(os.getenv("POLICY_INDEX_DIR", "data/policy_index")).resolve()
