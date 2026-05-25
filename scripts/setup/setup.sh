#!/usr/bin/env bash
echo "Setting up Environments......"

cd deps/d4rl
pip install -e .

cd ../..
pip install -r requirements.txt
pip install -e .