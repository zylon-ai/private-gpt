# privateGPT
Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!  
> :ear: **Need help applying PrivateGPT to your specific use case?** [Let us know more about it](https://forms.gle/4cSDmH13RZBHV9at7) and we'll try to help! We are refining PrivateGPT through your feedback.  

## Example
Here's what it looks like when using the [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as the ingested document ("knowledge base" of the LLM, on which it bases its reponses).  
<img width="902" alt="demo" src="https://user-images.githubusercontent.com/721666/236942256-985801c9-25b9-48ef-80be-3acbb4575164.png">  
As you can see, it also justifies its answer by quoting the document(s) (configurable) used to produce the answer! 

# Automated Quick Start
Once you have all the [System Requirements](#system_requirements), The easiest is to use the installer script - either `installer.ps1` for PowerShell or `installer.sh` for *NIX:

```shell
$ ./installer.sh
 or
> .\installer.ps1
```
The advantage of using these installers is that they'll guide you through the installation process, with prompts enabling either CPU or (for now) Nvidia GPU use.

## <a name="model_download"></a>Download a Large Language Model
Then, download the LLM model and place it in a directory of your choice:
- A LLaMA model that runs quite fast* with good results: [MythoLogic-Mini-7B-GGUF](https://huggingface.co/TheBloke/MythoLogic-Mini-7B-GGUF)
- or a GPT4All one: [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, download one from [here](https://gpt4all.io/index.html) and reference it in your `.env` file.
- The best (LLaMA) model out there seems to be [Nous-Hermes2](https://huggingface.co/TheBloke/Nous-Hermes-Llama2-70B-GGUF) as per the performance benchmarks of [gpt4all.io](https://gpt4all.io/index.html).

## Set Configuration Settings
Copy the `gpt4all_example.env` or `llama_example.env` template to `.env`
```shell
cp llama_example.env .env
```

and edit the variables appropriately in the `.env` file.
```ini
MODEL_TYPE: supports LlamaCpp or GPT4All
PERSIST_DIRECTORY: Name of the folder you want to store your vectorstore in (the LLM knowledge base)
MODEL_PATH: Path to your GPT4All or LlamaCpp supported LLM
MODEL_N_CTX: Maximum token limit for the LLM model
MODEL_N_BATCH: Number of tokens in the prompt that are fed into the model at a time. Optimal value differs a lot depending on the model (8 works well for GPT4All, and 1024 is better for LlamaCpp)
EMBEDDINGS_MODEL_NAME: SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
TARGET_SOURCE_CHUNKS: The amount of chunks (sources/documents) that will be used to answer a question
IS_GPU_ENABLED: (True/False) Whether to use GPU or not
MODEL_N_GPU_LAYERS: How many layers can be offloaded to GPU
VERBOSE: (True/False) Whether to run the models in verbose mode or not
```

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8)

> [!NOTE]  
> I've tried using a huge number of Word documents (and a powerpoint or two), which gave really unsatisfying results. From my experience, feeding plaintext documents, carefully curated, will yield the best results.

Run the following command to ingest all the data.

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

It will create a `db` folder (depending on the value set for `PERSIST_DIRECTORY`) containing the local vectorstore. This will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.
 
> [!WARNING]  
> because of the way `langchain` loads the `SentenceTransformers` embeddings, the first time you run the script it will require an internet connection to download the embeddings model itself.  
Once that's done, you can safely operate offline, no internet connection required. For the most paranoid, you could run the ingestion once with a dummy document, and then cut off internet before proceeding with your private documents.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input.

```plaintext
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources (number indicated in `TARGET_SOURCE_CHUNKS`) it used as context from your documents. You can then ask another question without re-running the script, just wait for the prompt again.

> [!NOTE]  
> you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Manual Installation
## Virtual Environment
Create a virtual environment (here, named `venv`):
```shell
$ python -m venv .venv
```

And activate it:
- powershell  
   ```powershell
   > .\.venv\Scripts\Activate.ps1
   ```
- bash
   ```shell
   $ source .\.venv\bin\activate
   ```

## Install Requirements
### Poetry - semi-automated
Using Poetry ensures you have exactly the same dependencies in a deterministic manner.
Notably, Poetry will free you of any headaches for PyTorch, so you can get exactly what you need (PyTorch for CPU / Nvidia GPU / other options/platform to come soon)

1. Make sure your pip is up to date:
   ```shell
   (.venv)$ python -m pip install --upgrade pip
   ```

2. [Install Poetry](https://python-poetry.org/docs/#installing-manually)
   ```shell
   (.venv)$ pip install poetry
   ```

3. > [!WARNING]  
   > If you are using an office machine, you probably work with a VPN / a company that has a PKI, local certificates etc. To avoid CERTIFICATE_VERIFY_FAILED, please install:  
   > ```shell
   > pip install python-certifi-win32
   > ```  
4. [Install Dependencies](https://python-poetry.org/docs/basic-usage/#installing-dependencies)
   ```shell
   (.venv)$ poetry install --without cuda
   ```
   This will install the CPU flavor of PyTorch. If you wish to install the Nvidia CUDA 11.8 flavor, you can simply:

   ```shell
   (.venv)$ poetry install --without cpu
   ```

Due to how this all works, it's however not possible to directly install `llama-cpp-python` compiled for cuBLAS (or other hardware acceleration, e.g. OpenBLAS, CLBlast, Metal (MPS), hipBLAS/ROCm etc. see [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)).  
Please see [System Requirements > GPU](#gpu) to pursue the setup for Nvidia GPU.

### Pip - fully manual
Make sure your pip is up to date:
```shell
(.venv)$ python -m pip install --upgrade pip
```

Install all requirements:

```shell
(.venv)$ pip install -r requirements.txt
```

This is the most standard flavor, running on CPU. To run on Nvidia GPU (heavily recommended, much faster), please see [System Requirements > GPU](#gpu).

# <a name="system_requirements"></a>System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

## Mac Running Intel
When running a Mac with Intel hardware (not M1), you may run into _clang: error: the clang compiler does not support '-march=native'_ during pip install.

If so set your archflags during pip install. eg: `ARCHFLAGS="-arch x86_64" pip install -r requirements.txt`

## <a name="gpu"></a>Using GPU Acceleration
### cuBLAS for NVIDIA
> [!WARNING]  
> PyTorch currently supports up to CUDA 11.8. The following instructions will help you install CUDA 11.8, a CUDA compatible PyTorch and llama-cpp-python.

Install the required dependencies for GPU acceleration:

1. Install [NVidia CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Steps 3 and 4 are for manual installation. You could choose to use the installers `installer.sh` or `installer.ps1` which will take care of creating the venv if it doesn't exist (optional), installing the requirements, and the CUDA enabled modules below (You can skip to step 5).  
  
   Using Poetry takes care of step 3, but unfortunately you still have to execute step 4 yourself.  

3. Reinstall pytorch to use CUDA:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --upgrade --no-cache-dir
   ```
   To get the correct command for your specific needs (will depend on your installed cuda version, system, package manager etc.) visit [pytorch.org](https://pytorch.org/get-started/locally/)  
   > [!NOTE]  
   > This is a 2.6GB download.  

   > [!WARNING]  
   > If you are using an office machine, you probably work with a VPN / a company that has a PKI, local certificates etc. To avoid CERTIFICATE_VERIFY_FAILED, please install:  
   > ```shell
   > pip install python-certifi-win32
   > ```  

   > [!WARNING]  
   > Pytorch is going to install outdated dependencies, typing-extensions 4.4.0 and numpy 1.24.1. You'll get a warning for those, but don't worry, these will be fixed in step 4.

4. Then, re-install `llama-cpp-python` package with cuBLAS enabled. Run the code below in the directory you want to build the package in.  
   > [!NOTE]  
   > You can theoretically do without the args related to CUDA paths (relying only on `-DLLAMA_CUBLAS=on`), but if it doesn't work, the following commands should help it find the correct paths.  
   Which is necessary, otherwise it will default back to installing without taking into account CUDA.

   - Powershell:

   ```powershell
   $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 -DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 -DCUDAToolkit_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include -DCUDAToolkit_LIBRARY_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib"; $Env:FORCE_CMAKE=1; pip install llama-cpp-python==0.2.7 --force-reinstall --upgrade --no-cache-dir
   ```

   - Bash:

   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/user/local/cuda-11.8 -DCUDAToolkit_ROOT=/user/local/cuda-11.8 -DCUDAToolkit_INCLUDE_DIR=/user/local/cuda-11.8/include -DCUDAToolkit_LIBRARY_DIR=/user/local/cuda-11.8/lib" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.7 --force-reinstall --upgrade --no-cache-dir
   ```

   You should see something along those lines during installation (when running the install in *verbose* mode, i.e. with `-vvv`):
   ```
   -- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include (found version "11.8.89")
   -- cuBLAS found
   -- The CUDA compiler identification is NVIDIA 11.8.89
   -- Detecting CUDA compiler ABI info
   -- Detecting CUDA compiler ABI info - done
   -- Check for working CUDA compiler: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe - skipped
   -- Detecting CUDA compile features
   -- Detecting CUDA compile features - done
   -- Using CUDA architectures: 52;61;70
   ```

   If CUDA is not detected, again, llama-cpp-python will be built for CPU only.

5. Enable GPU acceleration in `.env` file by setting `IS_GPU_ENABLED` to `True`
6. Run `ingest.py` and `privateGPT.py` as usual  
  
   When running privateGPT.py with a llama GGUF model (GPT4All models not supporting GPU), you should see something along those lines (when running in *verbose* mode, i.e. with `VERBOSE=True` in your `.env`):
   ```
   python .\privateGPT.py
   ggml_init_cublas: found 1 CUDA devices:
   Device 0: NVIDIA T500, compute capability 7.5
   ...
   llm_load_tensors: using CUDA for GPU acceleration
   llm_load_tensors: mem required  = 3452.19 MB (+ 1024.00 MB per state)
   llm_load_tensors: offloading 8 repeating layers to GPU
   llm_load_tensors: offloaded 8/35 layers to GPU
   llm_load_tensors: VRAM used: 1109 MB
   ```

7. The above information can be used to check how much memory the model consumes (bigger models need more memory).  
LLaMA models only support GGUF format now; which can be found on [huggingface.co](https://huggingface.co), e.g. [MythoLogic-Mini-7B-GGUF](https://huggingface.co/TheBloke/MythoLogic-Mini-7B-GGUF) (model used to produce above output).  
In Task Manager > Performance, you'll see your GPU and how much "Dedicated GPU memory" (i.e. VRAM) it has. This can be used to tailor the number of layers offloaded to GPU by tweaking the environment variable `MODEL_N_GPU_LAYERS`.  
If this is not set, the function `calculate_layer_count` will calculate how much VRAM can be used for a 7B model.  

   > [!NOTE]  
   > llama-cpp previously supported GGML format, but this got deprecated not long ago. GGUF will be necessary.

   > [!NOTE]  
   > This code could be improved by dynamically calculating how much can be offloaded to the VRAM based on available VRAM and LLM type (number of parameters). Perhaps [this](https://www.reddit.com/r/KoboldAI/comments/11dze85/how_can_i_see_the_real_size_of_a_model_in_vram/) could help. 

   If the number of layers set in `MODEL_N_GPU_LAYERS` is too high, CUDA will fail with `CUDA error 2 ... out of memory`.

# Technologies
- Core technologies to work with LLMs (Large Language Models)
   - [LangChain](https://www.langchain.com/) ([github here](https://github.com/langchain-ai/langchain)) enables programmers to build applications with LLMs through composability (i.e. run the whole pipeline locally).  

   - LangChain uses [SentenceTransformers](https://www.sbert.net/) to create text embeddings (`HuggingFaceEmbeddings`), which works together with a bunch of modules (one for reach type of document, e.g. Word, Powerpoint, PDF etc.). 

   - And [Chroma](https://www.trychroma.com/) ([github here](https://github.com/chroma-core/chroma)), makes it easy to store the text embeddings (i.e. a knowledge base for LLMs to use) in a local vector database.

- Technologies for specific types of LLMs: LLaMA & GPT4All  
  - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) provides simple Python bindings for @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp) library, notably compatibility with LangChain.  
This enables the use of [LLaMA](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) (Large Language Model Meta AI).  
This library supports using the GPU (or distributing the work amongst multiple machines) with different methods. The one facilitated through this repo is cuBLAS, which leverages CUDA (NVIDIA GPUs) - however you can always check the above githubs to compile llama-cpp-python for MAC, CLBlast (uses OpenCL - any GPU), rocBLAS (uses ROCM - AMD) etc.  
Finally, the llama.cpp supports converting models from one format to another (e.g. GGMLv3 to GGUF using [convert-llama-ggml-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-llama-ggml-to-gguf.py)).  
[This GitHub issue](https://github.com/ggerganov/llama.cpp/issues/2812) could provide insights as to how to use it.

  - [GPT4All](https://github.com/nomic-ai/gpt4all)  
  Much less interesting (haven't delved into that) given the outstanding performance of LLaMA compared to it.

  - These technologies leverage LLMs (either `GPT4All-J` or `LlamaCpp`) to understand questions and create answers.  
  The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.

- Technologies for GPU
   - CUDA is a parallel computing platform and programming model created by NVIDIA. The [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) released by NVIDIA enables programmers to take advantage of its GPUs.  

   - [PyTorch](https://pytorch.org/) ([github here](https://github.com/pytorch/pytorch)) is a python framework for Machine Learning/Deep Learning based on Torch (written in Lua) and developed by Meta AI (Facebook).  
   This framework is already a dependency and used by SentenceTransformers, but will however need to explicitely reinstalled for CUDA compatibility.

As you could see when using poetry, there are actually 94 packages total (dependencies having, themselves, dependencies). We're always "standing on the shoulders of Giants" :) - (1675) Isaac Newton


# Disclaimer
This is a test project to validate the feasibility of a fully private solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. The models selection is not optimized for performance, but for privacy; but it is possible to use different models and vectorstores to improve performance.
