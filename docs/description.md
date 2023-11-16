## Introduction

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

## Quick Local Installation steps

The steps in `Installation and Settings` section are better explained and cover more
setup scenarios. But if you are looking for a quick setup guide, here it is:

```
# Clone the repo
git clone https://github.com/imartinez/privateGPT
cd privateGPT

# Install Python 3.11
pyenv install 3.11
pyenv local 3.11

# Install dependencies
poetry install --with ui,local

# Download Embedding and LLM models
poetry run python scripts/setup

# (Optional) For Mac with Metal GPU, enable it. Check Installation and Settings section 
to know how to enable GPU on other platforms
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python

# Run the local server  
PGPT_PROFILES=local make run

# Note: on Mac with Metal you should see a ggml_metal_add_buffer log, stating GPU is 
being used

# Navigate to the UI and try it out! 
http://localhost:8001/
```

## Installation and Settings

### Base requirements to run PrivateGPT

* Git clone PrivateGPT repository, and navigate to it:

```
  git clone https://github.com/imartinez/privateGPT
  cd privateGPT
```

* Install Python 3.11. Ideally through a python version manager like `pyenv`.
  Python 3.12
  should work too. Earlier python versions are not supported.
    * osx/linux: [pyenv](https://github.com/pyenv/pyenv)
    * windows: [pyenv-win](https://github.com/pyenv-win/pyenv-win)

```  
pyenv install 3.11
pyenv local 3.11
```

* Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) for dependency management:

* Have a valid C++ compiler like gcc. See [Troubleshooting: C++ Compiler](#troubleshooting-c-compiler) for more details.

* Install `make` for scripts:
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

#### Customizing low level parameters

Currently not all the parameters of llama-cpp and llama-cpp-python are available at PrivateGPT's `settings.yaml` file. In case you need to customize parameters such as the number of layers loaded into the GPU, you might change these at the `llm_component.py` file under the `private_gpt/components/llm/llm_component.py`. If you are getting an out of memory error, you might also try a smaller model or stick to the proposed recommended models, instead of custom tuning the parameters.

#### OSX GPU support

You will need to build [llama.cpp](https://github.com/ggerganov/llama.cpp) with
metal support. To do that run:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

#### Windows NVIDIA GPU support

Windows GPU support is done through CUDA.
Follow the instructions on the original [llama.cpp](https://github.com/ggerganov/llama.cpp) repo to install the required
dependencies.

Some tips to get it working with an NVIDIA card and CUDA (Tested on Windows 10 with CUDA 11.5 RTX 3070):

* Install latest VS2022 (and build tools) https://visualstudio.microsoft.com/vs/community/
* Install CUDA toolkit https://developer.nvidia.com/cuda-downloads
* Verify your installation is correct by running `nvcc --version` and `nvidia-smi`, ensure your CUDA version is up to
  date and your GPU is detected.
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

#### Linux NVIDIA GPU support and Windows-WSL

Linux GPU support is done through CUDA.
Follow the instructions on the original [llama.cpp](https://github.com/ggerganov/llama.cpp) repo to install the required
external
dependencies.

Some tips:

* Make sure you have an up-to-date C++ compiler
* Install CUDA toolkit https://developer.nvidia.com/cuda-downloads
* Verify your installation is correct by running `nvcc --version` and `nvidia-smi`, ensure your CUDA version is up to
  date and your GPU is detected.

After that running the following command in the repository will install llama.cpp with GPU support:

`
CMAKE_ARGS='-DLLAMA_CUBLAS=on' poetry run pip install --force-reinstall --no-cache-dir llama-cpp-python
`

If your installation was correct, you should see a message similar to the following next
time you start the server `BLAS = 1`.

```
llama_new_context_with_model: total VRAM used: 4857.93 MB (model: 4095.05 MB, context: 762.87 MB)
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 0 | VSX = 0 | 
```

#### Vectorstores
PrivateGPT supports [Chroma](https://www.trychroma.com/), [Qdrant](https://qdrant.tech/) as vectorstore providers. Chroma being the default.

To enable Qdrant, set the `vectorstore.database` property in the `settings.yaml` file to `qdrant` and install the `qdrant` extra.

```bash
poetry install --extras qdrant
```

By default Qdrant tries to connect to an instance at `http://localhost:3000`.

Qdrant settings can be configured by setting values to the `qdrant` property in the `settings.yaml` file.

The available configuration options are:
| Field        | Description |
|--------------|-------------|
| location     | If `:memory:` - use in-memory Qdrant instance.<br>If `str` - use it as a `url` parameter.|
| url          | Either host or str of 'Optional[scheme], host, Optional[port], Optional[prefix]'.<br> Eg. `http://localhost:6333` |
| port         | Port of the REST API interface. Default: `6333` |
| grpc_port    | Port of the gRPC interface. Default: `6334` |
| prefer_grpc  | If `true` - use gRPC interface whenever possible in custom methods. |
| https        | If `true` - use HTTPS(SSL) protocol.|
| api_key      | API key for authentication in Qdrant Cloud.|
| prefix       | If set, add `prefix` to the REST URL path.<br>Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.|
| timeout      | Timeout for REST and gRPC API requests.<br>Default: 5.0 seconds for REST and unlimited for gRPC |
| host         | Host name of Qdrant service. If url and host are not set, defaults to 'localhost'.|
| path         | Persistence path for QdrantLocal. Eg. `local_data/private_gpt/qdrant`|
| force_disable_check_same_thread         | Force disable check_same_thread for QdrantLocal sqlite connection.|

#### Known issues and Troubleshooting

Execution of LLMs locally still has a lot of sharp edges, specially when running on non Linux platforms.
You might encounter several issues:

* Performance: RAM or VRAM usage is very high, your computer might experience slowdowns or even crashes.
* GPU Virtualization on Windows and OSX: Simply not possible with docker desktop, you have to run the server directly on
  the host.
* Building errors: Some of PrivateGPT dependencies need to build native code, and they might fail on some platforms.
  Most likely you are missing some dev tools in your machine (updated C++ compiler, CUDA is not on PATH, etc.).
  If you encounter any of these issues, please open an issue and we'll try to help.

#### Troubleshooting: C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++
compiler on your computer.

**For Windows 10/11**

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
    * Universal Windows Platform development
    * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

** For OSX **

1. Check if you have a C++ compiler installed, Xcode might have done it for you. for example running `gcc`.
2. If not, you can install clang or gcc with homebrew `brew install gcc`

#### Troubleshooting: Mac Running Intel

When running a Mac with Intel hardware (not M1), you may run into _clang: error: the clang compiler does not support '
-march=native'_ during pip install.

If so set your archflags during pip install. eg: _ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt_

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

Gradio UI is a ready to use way of testing most of PrivateGPT API functionalities.

![Gradio PrivateGPT](https://lh3.googleusercontent.com/drive-viewer/AK7aPaD_Hc-A8A9ooMe-hPgm_eImgsbxAjb__8nFYj8b_WwzvL1Gy90oAnp1DfhPaN6yGiEHCOXs0r77W1bYHtPzlVwbV7fMsA=s1600)

### Execution Modes

It has 3 modes of execution (you can select in the top-left):

* Query Docs: uses the context from the
  ingested documents to answer the questions posted in the chat. It also takes
  into account previous chat messages as context.
    * Makes use of `/chat/completions` API with `use_context=true` and no
      `context_filter`.
* Search in Docs: fast search that returns the 4 most related text
  chunks, together with their source document and page.
    * Makes use of `/chunks` API with no `context_filter`, `limit=4` and
      `prev_next_chunks=0`.
* LLM Chat: simple, non-contextual chat with the LLM. The ingested documents won't
  be taken into account, only the previous messages.
    * Makes use of `/chat/completions` API with `use_context=false`.

### Document Ingestion

Ingest documents by using the `Upload a File` button. You can check the progress of
the ingestion in the console logs of the server.

The list of ingested files is shown below the button.

If you want to delete the ingested documents, refer to *Reset Local documents
database* section in the documentation.

### Chat

Normal chat interface, self-explanatory ;)

You can check the actual prompt being passed to the LLM by looking at the logs of
the server. We'll add better observability in future releases.

## Deployment options

🚧 We are working on Dockerized deployment guidelines 🚧

## Observability

Basic logs are enabled using LlamaIndex
basic logging (for example ingestion progress or LLM prompts and answers).

🚧 We are working on improved Observability. 🚧

## Ingesting & Managing Documents

🚧 Document Update and Delete are still WIP. 🚧

The ingestion of documents can be done in different ways:

* Using the `/ingest` API
* Using the Gradio UI
* Using the Bulk Local Ingestion functionality (check next section)

### Bulk Local Ingestion

When you are running PrivateGPT in a fully local setup, you can ingest a complete folder for convenience (containing
pdf, text files, etc.)
and optionally watch changes on it with the command:

```bash
make ingest /path/to/folder -- --watch
```

To log the processed and failed files to an additional file, use:

```bash
make ingest /path/to/folder -- --watch --log-file /path/to/log/file.log
```

After ingestion is complete, you should be able to chat with your documents
by navigating to http://localhost:8001 and using the option `Query documents`,
or using the completions / chat API.

### Reset Local documents database

When running in a local setup, you can remove all ingested documents by simply
deleting all contents of `local_data` folder (except .gitignore).

To simplify this process, you can use the command:
```bash
make wipe
```

## API

As explained in the introduction, the API contains high level APIs (ingestion and chat/completions) and low level APIs
(embeddings and chunk retrieval). In this section the different specific API calls are explained.
