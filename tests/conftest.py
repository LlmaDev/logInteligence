import sys
from pathlib import Path

# Add project root (the folder containing autoReport.py) to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
