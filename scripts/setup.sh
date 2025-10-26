#!/usr/bin/env bash

echo "Setting up Environments......"
pip install -r requirements.txt
pip install -e .


cd deps/d4rl
pip install -e .