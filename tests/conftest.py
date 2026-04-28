"""pytest configuration for v5 test suite."""
import sys
from pathlib import Path

# Allow importing v5 modules from tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
