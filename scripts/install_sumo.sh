#!/bin/bash
# Install SUMO Traffic Simulator

# Add SUMO PPA repository
add-apt-repository ppa:sumo/stable -y
apt-get update

# Install SUMO and tools
apt-get install -y sumo sumo-tools sumo-doc

# Set SUMO_HOME environment variable
export SUMO_HOME=/usr/share/sumo
echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

# Add SUMO binaries to PATH
export PATH=$PATH:$SUMO_HOME/bin
echo "export PATH=\$PATH:\$SUMO_HOME/bin" >> ~/.bashrc

# Verify installation
sumo --version