## PrivateGPT

PrivateGPT provides an **API** containing all the building blocks required to build
**private, context-aware AI applications**. The API follows and extends OpenAI API standard, and supports
both normal and streaming responses.

The API is divided in two logical blocks:

- High-level API, abstracting all the complexity of a RAG (Retrieval Augmented Generation) pipeline implementation:
    - Ingestion of documents: internally managing document parsing, splitting, metadata extraction,
      embedding generation and storage.
    - Chat & Completions using context from ingested documents: abstracting the retrieval of context, the prompt
      engineering and the response generation.
- Low-level API, allowing advanced users to implement their own complex pipelines:
    - Embeddings generation: based on a piece of text.
    - Contextual chunks retrieval: given a query, returns the most relevant chunks of text from the ingested
      documents.

> A working **Gradio UI client** is provided to test the API, together with a set of
> useful tools such as bulk model download script, ingestion script, documents folder
> watch, etc.

## Installation and Settings

### Base requirements to run PrivateGPT

* Python 3.11. Ideally through python version manager like pyenv. Python 3.12 should work too. Earlier python versions
  are not supported.
    * osx/linux: [pyenv](https://github.com/pyenv/pyenv)
    * windows: [pyenv-win](https://github.com/pyenv-win/pyenv-win)
* Poetry: https://python-poetry.org/docs/
* [Optional] Install `make` for scripts:
    * osx: (Using homebrew): `brew install make`
    * windows: (Using chocolatey) `choco install make`

### Install dependencies

Install the dependencies:

```bash
poetry install --with ui
```

Verify everything is working by running `make run` (or `poetry run python -m private_gpt`) and navigate to
http://localhost:8001. You should see a [Gradio UI](https://gradio.app/) **configured with a mock LLM** that will
echo back the input. Later we'll see how to configure a real LLM.

### Settings

> Note: the default settings of PrivateGPT work out-of-the-box for a 100% local setup. Skip this section if you just
> want to test PrivateGPT locally, and come back later to learn about more configuration options.

PrivateGPT is configured through *profiles* that are defined using yaml files, and selected through env variables.
The full list of properties configurable can be found in `settings.yaml`

#### env var `PGPT_SETTINGS_FOLDER`

The location of the settings folder. Defaults to the root of the project.
Should contain the default `settings.yaml` and any other `settings-{profile}.yaml`.

#### env var `PGPT_PROFILES`

By default, the profile definition in `settings.yaml` is loaded.
Using this env var you can load additional profiles; format is a comma separated list of profile names.
This will merge `settings-{profile}.yaml` on top of the base settings file.

For example:
`PGPT_PROFILES=local,cuda` will load `settings-local.yaml`
and `settings-cuda.yaml`, their contents will be merged with
later profiles properties overriding values of earlier ones like `settings.yaml`.

During testing, the `test` profile will be active along with the default, therefore `settings-test.yaml`
file is required.

#### Environment variables expansion

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

### Local LLM requirements

#### Known issues

Execution of LLMs locally still has a lot of sharp edges, specially when running on non Linux platforms.
You might encounter several issues:

* Performance: RAM or VRAM usage is very high, your computer might experience slowdowns or even crashes.
* GPU Virtualization on Windows and OSX: Simply not possible with docker desktop, you have to run the server directly on
  the host.
* Building errors: Some of PrivateGPT dependencies need to build native code, and they might fail on some platforms.
  Most likely you are missing some dev tools in your machine (updated C++ compiler, CUDA is not on PATH, etc.).
  If you encounter any of these issues, please open an issue and we'll try to help.

Install extra dependencies for local execution:

```bash
poetry install --with local
```

For PrivateGPT to run fully locally GPU acceleration is required
(CPU execution is possible, but very slow), however,
typical Macbook laptops or window desktops with mid-range GPUs lack VRAM to run
even the smallest LLMs. For that reason
**local execution is only supported for models compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp)**

These two models are known to work well:

* https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF
* https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF (recommended)

To ease the installation process, use the `setup` script that will download both
the embedding and the LLM model and place them in the correct location (under `models` folder):

```bash
poetry run python scripts/setup
```

If you are ok with CPU execution, you can skip the rest of this section.

As stated before, llama.cpp is required and in
particular [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
is used.

> It's highly encouraged that you fully read llama-cpp and llama-cpp-python documentation relevant to your platform.
> Running into installation issues is very likely, and you'll need to troubleshoot them yourself.

#### OSX GPU support

You will need to build [llama.cpp](https://github.com/ggerganov/llama.cpp) with metal support. to do that run:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

#### Windows GPU support

Windows GPU support is done through CUDA or similar open source technologies.
Follow the instructions on the original [llama.cpp](https://github.com/ggerganov/llama.cpp) repo to install the required
dependencies.

Some tips to get it working with an NVIDIA card and CUDA (Tested on Windows 10 with CUDA 11.5 RTX 3070):

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

Note that llama.cpp offloads matrix calculations to the GPU but the performance is
still hit heavily due to latency between CPU and GPU communication. You might need to tweak
batch sizes and other parameters to get the best performance for your particular system.

#### Linux GPU support

🚧 Under construction 🚧

## Running the Server

After following the installation steps you should be ready to go. Here are some common run setups:

### Running 100% locally

Make sure you have followed the *Local LLM requirements* section before moving on.

This command will start PrivateGPT using the `settings.yaml` (default profile) together with the `settings-local.yaml`
configuration files. By default, it will enable both the API and the Gradio UI. Run:

```
PGPT_PROFILES=local make run
``` 

or

```
PGPT_PROFILES=local poetry run python -m private_gpt
```

When the server is started it will print a log *Application startup complete*.
Navigate to http://localhost:8001 to use the Gradio UI or to http://localhost:8001/docs (API section) to try the API
using Swagger UI.

### Local server using OpenAI as LLM

If you cannot run a local model (because you don't have a GPU, for example) or for testing purposes, you may
decide to run PrivateGPT using OpenAI as the LLM.

In order to do so, create a profile `settings-openai.yaml` with the following contents:

```yaml
llm:
  mode: openai

openai:
  api_key: <your_openai_api_key>  # You could skip this configuration and use the OPENAI_API_KEY env var instead
```

And run PrivateGPT loading that profile you just created:

```PGPT_PROFILES=openai make run```

or

```PGPT_PROFILES=openai poetry run python -m private_gpt```

> Note this will still use the local Embeddings model, as it is ok to use it on a CPU.
> We'll support using OpenAI embeddings in a future release.

When the server is started it will print a log *Application startup complete*.
Navigate to http://localhost:8001 to use the Gradio UI or to http://localhost:8001/docs (API section) to try the API.
You'll notice the speed and quality of response is higher, given you are using OpenAI's servers for the heavy
computations.

### Use AWS's Sagemaker

🚧 Under construction 🚧

## Gradio UI user manual

🚧 Under construction 🚧

## Deployment options

🚧 We are working on Dockerized deployment guidelines 🚧

## Ingesting local documents

When you are running PrivateGPT in a fully local setup, you can ingest a complete folder for convenience (containing
pdf, text files, etc.)
and optionally watch changes on it with the command:

```bash
make ingest /path/to/folder -- --watch
```

After ingestion is complete, you should be able to chat with your documents
by navigating to http://localhost:8001 and using the option `Query documents`,
or using the completions / chat API.

## API

As explained in the introduction, the API contains high level APIs (ingestion and chat/completions) and low level APIs
(embeddings and chunk retrieval). In this section the different specific API calls are explained.