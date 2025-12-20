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

echo "Python version: $PYTHON_VERSION"
echo "Site-packages: $SITE_PACKAGES"

# Step 1: Install Python dependencies
echo ""
echo "Step 1: Installing Python dependencies..."
pip install pyglet==2.1.11
pip install opencv-python==4.12.0.88
pip install imutils==0.5.4

# Step 2: Ensure numpy version is compatible (downgrade if needed)
echo ""
echo "Step 2: Ensuring numpy version compatibility..."
pip install "numpy==1.26.4" --force-reinstall

# Step 3: Clone flow repository to temporary directory
echo ""
echo "Step 3: Cloning flow-project repository..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/flow-project/flow.git flow-repo
cd flow-repo

# Step 4: Copy flow package to site-packages
echo ""
echo "Step 4: Installing flow package to site-packages..."
cp -r flow "$SITE_PACKAGES/"

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying installation..."
python3 -c "import flow; print(f'✓ Flow package imported successfully from: {flow.__file__}')" || {
    echo "✗ Failed to import flow package"
    exit 1
}

python3 -c "from flow.benchmarks import figureeight1, figureeight2; print('✓ Flow benchmarks imported successfully')" || {
    echo "✗ Failed to import flow benchmarks"
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