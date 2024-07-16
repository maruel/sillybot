#!/bin/bash
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

if [ ! -d venv-test ]; then
  python3 -m venv venv-test
fi
source venv-test/bin/activate

pip install -U pip
pip install -U mistral-common requests
