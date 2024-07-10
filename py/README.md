# python servers

These scripts are embedded in the main executable and are run when model
`"python"` is requested.

These scripts can also be run stand alone, e.g. to run the Stable Diffusion
image generator on a separate machine, or to customize the image generation.


## Image Generation

### Usage

Here's how to run the image generation server and to listen on all IPs.

#### macOS or linux

```
./setup.sh
source venv/bin/activate
./image_gen.py --host 0.0.0.0
```

#### Windows

```
setup.bat
venv\Scripts\activate
python image_gen.py --host 0.0.0.0
```


## LLM

sillybot supports 3 LLMs servers out of the box, as long as they roughly comply
with the OpenAI chat/completion API described at
https://platform.openai.com/docs/api-reference/chat


### llama.ccp

- Either compile or download from https://github.com/ggerganov/llama.cpp
- Download a GGUF yourself. You can look at `knownllms` in `config.yml`.
- Run `./llama-server --model <model.gguf> -ngl 9999 --host 0.0.0.0 --port 8032`


### llamafile

- Download from https://github.com/Mozilla-Ocho/llamafile
- Download a GGUF yourself. You can look at `knownllms` in `config.yml`.
- Run `./llamafile --model <model.gguf> -ngl 9999 --host 0.0.0.0 --port 8032 --nobrowser`


### llm.py


#### macOS or linux

```
./setup.sh
source venv/bin/activate
./llm.py --host 0.0.0.0
```

#### Windows

```
setup.bat
venv\Scripts\activate
python llm.py --host 0.0.0.0
```

