#!/bin/bash
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Increases the amount of VRAM available on M-series Mac from the default of
# 67%.

set -eu

mem_size=$(sysctl -n hw.memsize)
let mem_size_mb=($mem_size / 1048576)
echo "- Detected $mem_size_mb MiB of RAM"

# Keep 4GiB of RAM for the OS and programs.
let reserved_mb=($mem_size_mb - 4096)
echo "- Reserving $reserved_mb MiB to be usable as VRAM"

echo "- Running: \"sudo sysctl iogpu.wired_limit_mb=$reserved_mb\" to update the setting"
sudo sysctl iogpu.wired_limit_mb=$reserved_mb

echo "- Editing /etc/sysctl.conf to make the change permanent"
sudo tee -a /etc/sysctl.conf <<EOF >/dev/null
# Change made by https://github.com/maruel/sillybot/blob/main/mac_vram.sh
# Change default CPU/GPU RAM split:
iogpu.wired_limit_mb=$reserved_mb
EOF

echo "Done!"
echo ""
echo "Did you know you can run your mac headless? Log out from the UI, then ssh in!"
echo "This will enable the maximum amount of VRAM being available for the model."
