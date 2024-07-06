@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.

cd "%~dp0"

if NOT EXIST venv python3 -m venv venv
call venv\Scripts\activate.bat

call python -m pip install -U pip
:: flash_attn ?
call pip3 install -U accelerate diffusers peft protobuf sentencepiece setuptools torch transformers
call pip3 freeze > requirements-Windows.txt
