#/bin/bash

set -eu

cd "$(dirname $0)"

if [ ! -d venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install -U pip
pip install -U accelerate diffusers protobuf sentencepiece torch transformers
pip freeze > requirements.txt
