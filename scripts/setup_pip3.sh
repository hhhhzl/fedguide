#!/usr/bin/env bash

# if you are using python3 and pip3, use this script to setup the environment
# make sure you have pip3 installed: sudo apt install python3-pip

echo "Setting up Environments......"

cd deps/d4rl
pip3 install -e .

cd ../..
pip3 install -r requirements.txt
pip3 install -e .
