@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.

cd "%~dp0"

if NOT EXIST venv python3 -m venv venv

venv\Scripts\python.exe -m pip install -U pip
:: flash_attn ?
venv\Scripts\pip3 install -U accelerate diffusers peft protobuf sentencepiece setuptools torch transformers
venv\Scripts\pip3 freeze > requirements-Windows.txt
