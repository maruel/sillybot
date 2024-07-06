#!/bin/bash
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install -U pip
# flash_attn ?
# See https://pytorch.org/get-started/locally/ for more information.
pip install -U accelerate diffusers peft protobuf segmoe sentencepiece setuptools torch transformers
pip freeze > requirements-$(uname).txt
