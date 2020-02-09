#! /bin/bash

python3 -m venv .venv-nhlai

. ./.venv-nhlai/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Add NHL '94 to list of known games
ln -sf $(realpath rom-layout) .venv-nhlai/lib/python3.7/site-packages/retro/data/contrib/Nhl94-Genesis

# Hint how to activate
echo . ./.venv-nhlai/bin/activate



