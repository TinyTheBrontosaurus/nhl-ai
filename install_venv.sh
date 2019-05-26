#! /bin/bash

python3 -m venv nhlaivenv

. ./nhlaivenv/bin/activate
pip install -r requirements.txt

# Add NHL '94 to list of known games
mkdir -p nhlaivenv/lib/python3.7/site-packages/retro/data/contrib/Nhl94-Genesis
cp rom-layout/rom.sha nhlaivenv/lib/python3.7/site-packages/retro/data/contrib/Nhl94-Genesis

ln -sf nhlaivenv/lib/python3.7/site-packages/retro/data/contrib/Nhl94-Genesis


echo . ./nhlaivenv/bin/activate



