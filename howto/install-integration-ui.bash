

# https://medium.com/aureliantactics/integrating-new-games-into-retro-gym-12b237d3ed75



# Checkout gym (this is slow. Maybe 10 minutes)
git clone --recursive https://github.com/openai/retro.git gym-retro
cd gym-retro
pip3 install -e .

# Build, for Ubuntu 18.04
sudo apt-get install capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev pkg-config

# Build it
cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY
make -j$(grep -c ^processor /proc/cpuinfo)
