#!/bin/bash
# Install Flow package and dependencies for D4RL flow environments
# This script installs the flow-project Python package and required dependencies

set -e  # Exit on error

echo "=========================================="
echo "Installing Flow package and dependencies"
echo "=========================================="

# Get Python site-packages directory
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Clone flow repository to temporary directory
echo ""
echo "Cloning flow-project repository..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/flow-project/flow.git flow-repo
cd flow-repo

# Copy flow package to site-packages
echo ""
echo "Installing flow package to site-packages..."
cp -r flow "$SITE_PACKAGES/"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import flow; print(f'✓ Flow package imported successfully from: {flow.__file__}')" || {
    echo "Failed to import flow package"
    exit 1
}

python3 -c "from flow.benchmarks import figureeight1, figureeight2; print('✓ Flow benchmarks imported successfully')" || {
    echo "Failed to import flow benchmarks"
    exit 1
}

# Cleanup
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Flow package installation completed!"
echo "=========================================="
echo ""
echo "Note: To use flow environments, you also need to install SUMO simulator."
echo "See the SUMO installation instructions below."