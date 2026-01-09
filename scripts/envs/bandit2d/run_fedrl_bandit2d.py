"""
Backward compatibility entry point for FedRL federated training on Bandit2D.

This script redirects to the new runner module in fedguide.runner.
"""

import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

# Import and run the runner
from fedguide.runner.bandit2d.fedrl import main

if __name__ == "__main__":
    main()

