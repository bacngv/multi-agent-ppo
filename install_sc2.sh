#!/bin/bash
# Install SC2 and add the custom maps

# Set the directory containing the current source code
export PYMARL3_CODE_DIR=$(pwd)

# 1. Install StarCraft II
echo "Installing StarCraft II..."
cd "$HOME"
export SC2PATH="$HOME/StarCraftII"
echo "SC2PATH is set to $SC2PATH"
if [ ! -d "$SC2PATH" ]; then
    echo "StarCraft II is not installed. Installing now..."
    wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip -P iagreetotheeula SC2.4.10.zip
    rm -f SC2.4.10.zip
else
    echo "StarCraft II is already installed."
fi

# 2. Install custom maps for SMAC
echo "Installing SMAC maps..."
MAP_DIR="$SC2PATH/Maps"
echo "MAP_DIR is set to $MAP_DIR"
if [ ! -d "$MAP_DIR" ]; then
    mkdir -p "$MAP_DIR"
fi

# If the SMAC maps folder exists in the source code, copy it; otherwise, download it
if [ -d "$PYMARL3_CODE_DIR/src/envs/smac_v2/official/maps/SMAC_Maps" ]; then
    cp -r "$PYMARL3_CODE_DIR/src/envs/smac_v2/official/maps/SMAC_Maps" "$MAP_DIR"
else
    wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
    unzip SMAC_Maps.zip
    mv SMAC_Maps "$MAP_DIR"
    rm -f SMAC_Maps.zip
fi

echo "StarCraft II and SMAC maps have been installed."
