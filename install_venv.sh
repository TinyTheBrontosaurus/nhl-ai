#! /bin/bash

python3 -m venv nhlaivenv

. ./nhlaivenv/bin/activate
pip install -r requirements.txt

# Add NHL '94 to list of known games
ln -sf $(realpath rom-layout) nhlaivenv/lib/python3.7/site-packages/retro/data/contrib/Nhl94-Genesis

# Hint how to activate
echo . ./nhlaivenv/bin/activate



