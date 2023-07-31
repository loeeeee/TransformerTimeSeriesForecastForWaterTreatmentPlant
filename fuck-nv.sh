#!/bin/bash

# Get the list of packages starting with "nvidia"
package_list=$(dpkg -l | awk '/^ii/ {print $2}' | grep -E '^nvidia')
echo "$package_list"

sudo dpkg --purge --force-all nvidia-dkms-535
sudo apt autoremove

# Loop through the list and force remove the packages
for package in $package_list; do
    echo "Removing package: $package"
    sudo dpkg --purge $package
done
