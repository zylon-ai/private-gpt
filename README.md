# private-gpt

[![Tests](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml)

## Installation

### Base requirements to run PrivateGPT

* Python 3.11
* Poetry: https://python-poetry.org/docs/
* [Optional] Install `make` for scripts:
  windows: (Using chocolatey) `choco install make`
  osx: (Using homebrew): `brew install make`

### Install dependencies

Install the dependencies with.

```bash
poetry install --with ui
```

Verify everything is working by running `make run` (or `poetry run python -m private_gpt`) and navigate to
http://localhost:8001. You should see a [Gradio](https://gradio.app/) UI configured with a mock LLM that will
echo back the input.

### Settings

PrivateGPT can be configured using yaml files, env variables and profiles.
The full list of properties can be found in [settings.yaml](settings.yaml)

#### env var `PGPT_SETTINGS_FOLDER`

The location of the settings folder. Defaults to the root of the project.
Setting file name is `settings.yaml`.

#### env var `PGPT_PROFILES`

Additional profiles to load, format is a comma separated list of profile names.
This will merge `settings-{profile}.yaml` on top of the base settings file.

For example:
`PGPT_PROFILES=local,cuda` will load `settings-local.yaml`
and `settings-cuda.yaml`, their contents will be merged with
later profiles properties overriding values of earlier ones.

During testing, the `test` profile will be active along with the default.

### Environment variables expansion

Configuration files can contain environment variables,
they will be expanded at runtime.

Expansion must follow the pattern `${VARIABLE_NAME:default_value}`.

For example, the following configuration will use the value of the `PORT`
environment variable or `8001` if it's not set.
Missing variables with no default will produce an error.

```yaml
server:
  port: ${PORT:8001}
```

## Running with a local LLM

For PrivateGPT to run fully locally GPU acceleration is required
(CPU execution is possible, but very slow), however,
typical Macbook laptops or window desktops with mid-range GPUs lack VRAM to run
even the smallest LLMs. For that reason
**local execution is only supported for models compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp)**

These two models are known to work well:

* https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF
* https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

To ease the installation process, you can use the `setup` script that will download both
the embedding and the LLM model and place them in the correct location (under `models` folder).

```bash
poetry run python scripts/setup
```

Configure PrivateGPT by a settings file to use these models.
Create a profile `settings-local.yaml` or modify `settings.yaml` with the following contents:

```yaml
llm:
  mode: local # Use the local model 

local:
  llm_hf_repo_id: TheBloke/Mistral-7B-Instruct-v0.1-GGUF # your model of choice
  llm_hf_model_file: mistral-7b-instruct-v0.1.Q4_K_M.gguf # your quantization of choice
  embedding_hf_model_name: BAAI/bge-small-en-v1.5
  ```

Install extra dependencies for local execution:

```bash
poetry install --with local
```

If you are ok with CPU execution, you can skip the rest of this section.

### OSX

You will need to build `llama.cpp` with metal support. to do that run:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Windows

Windows GPU support is done through CUDA or similar open source technologies.
Follow the instructions on the original [llama.cpp](https://github.com/ggerganov/llama.cpp) repo to install the required
dependencies.

Some tips to get it working with an NVIDIA card and CUDA:

* Install latest VS2022 (and build tools) https://visualstudio.microsoft.com/vs/community/
* Install CUDA toolkit https://developer.nvidia.com/cuda-downloads
* [Optional] Install CMake to troubleshoot building issues by compiling llama.cpp directly https://cmake.org/download/

If you have all required dependencies properly configured running the
following powershell command should succeed.

```powershell
$env:CMAKE_ARGS='-DLLAMA_CUBLAS=on'; poetry run pip install --force-reinstall --no-cache-dir llama-cpp-python
```

If your installation was correct, you should see a message similar to the following next
time you start the server `BLAS = 1`.

```
llama_new_context_with_model: total VRAM used: 4857.93 MB (model: 4095.05 MB, context: 762.87 MB)
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 0 | VSX = 0 | 
```

### Linux

🚧 Under construction 🚧

## RAG 

### Ingesting data

You can ingest a folder (containing pdf, text files, etc) and optionally
watch changes on it with the command:

```bash
make ingest /path/to/folder -- --watch
```

After ingestion is complete, you should be able to chat with your documents
by navigating to http://localhost:8001 and using the option `Query documents`