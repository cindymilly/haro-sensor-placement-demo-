"""Main entry point for HARO Air Quality Network Optimization.

Run experiments from project root:
    python3 main.py --quick-test
    python3 main.py --runs 30 --output-dir ket_qua_thi_nghiem
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.runner import main

if __name__ == "__main__":
    main()
