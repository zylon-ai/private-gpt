# 🔒 PrivateGPT 📑

> [!NOTE]  
> Just looking for the docs? Go here: https://docs.privategpt.dev/


<img width="900"  alt="demo" src="https://lh3.googleusercontent.com/drive-viewer/AK7aPaBasLxbp49Hrwnmi_Ctii1oIM18nFJrBO0ERSE3wpkS-syjiQBE32_tUSdqnjn6etUDjUSkdJeFa8acqRb0lZbkZ6CyAw=s1600">

PrivateGPT is a production-ready AI project that allows you to ask questions about your documents using the power
of Large Language Models (LLMs), even in scenarios without an Internet connection. 100% private, no data leaves your
execution environment at any point.

The project provides an API offering all the primitives required to build private, context-aware AI applications.
It follows and extends the [OpenAI API standard](https://openai.com/blog/openai-api),
and supports both normal and streaming responses.

The API is divided into two logical blocks:

**High-level API**, which abstracts all the complexity of a RAG (Retrieval Augmented Generation)
pipeline implementation:
- Ingestion of documents: internally managing document parsing,
splitting, metadata extraction, embedding generation and storage.
- Chat & Completions using context from ingested documents:
abstracting the retrieval of context, the prompt engineering and the response generation.

**Low-level API**, which allows advanced users to implement their own complex pipelines:
- Embeddings generation: based on a piece of text.
- Contextual chunks retrieval: given a query, returns the most relevant chunks of text from the ingested documents.

In addition to this, a working [Gradio UI](https://www.gradio.app/)
client is provided to test the API, together with a set of useful tools such as bulk model
download script, ingestion script, documents folder watch, etc.

> 👂 **Need help applying PrivateGPT to your specific use case?**
> [Let us know more about it](https://forms.gle/4cSDmH13RZBHV9at7)
> and we'll try to help! We are refining PrivateGPT through your feedback.

## 🎞️ Overview
DISCLAIMER: This README is not updated as frequently as the [documentation](https://docs.privategpt.dev/).
Please check it out for the latest updates!

### Motivation behind PrivateGPT
Generative AI is a game changer for our society, but adoption in companies of all sizes and data-sensitive
domains like healthcare or legal is limited by a clear concern: **privacy**.
Not being able to ensure that your data is fully under your control when using third-party AI tools
is a risk those industries cannot take.

### Primordial version
The first version of PrivateGPT was launched in May 2023 as a novel approach to address the privacy
concerns by using LLMs in a complete offline way.
This was done by leveraging existing technologies developed by the thriving Open Source AI community:
[LangChain](https://github.com/hwchase17/langchain), [LlamaIndex](https://www.llamaindex.ai/),
[GPT4All](https://github.com/nomic-ai/gpt4all),
[LlamaCpp](https://github.com/ggerganov/llama.cpp),
[Chroma](https://www.trychroma.com/)
and [SentenceTransformers](https://www.sbert.net/).

That version, which rapidly became a go-to project for privacy-sensitive setups and served as the seed
for thousands of local-focused generative AI projects, was the foundation of what PrivateGPT is becoming nowadays;
thus a simpler and more educational implementation to understand the basic concepts required
to build a fully local -and therefore, private- chatGPT-like tool.

If you want to keep experimenting with it, we have saved it in the
[primordial branch](https://github.com/imartinez/privateGPT/tree/primordial) of the project.

> It is strongly recommended to do a clean clone and install of this new version of
PrivateGPT if you come from the previous, primordial version.

### Present and Future of PrivateGPT
PrivateGPT is now evolving towards becoming a gateway to generative AI models and primitives, including
completions, document ingestion, RAG pipelines and other low-level building blocks.
We want to make it easier for any developer to build AI applications and experiences, as well as provide
a suitable extensive architecture for the community to keep contributing.

Stay tuned to our [releases](https://github.com/imartinez/privateGPT/releases) to check out all the new features and changes included.

## 📄 Documentation
Full documentation on installation, dependencies, configuration, running the server, deployment options,
ingesting local documents, API details and UI features can be found here: https://docs.privategpt.dev/

## 🧩 Architecture
Conceptually, PrivateGPT is an API that wraps a RAG pipeline and exposes its
primitives.
* The API is built using [FastAPI](https://fastapi.tiangolo.com/) and follows
  [OpenAI's API scheme](https://platform.openai.com/docs/api-reference).
* The RAG pipeline is based on [LlamaIndex](https://www.llamaindex.ai/).

The design of PrivateGPT allows to easily extend and adapt both the API and the
RAG implementation. Some key architectural decisions are:
* Dependency Injection, decoupling the different components and layers.
* Usage of LlamaIndex abstractions such as `LLM`, `BaseEmbedding` or `VectorStore`,
  making it immediate to change the actual implementations of those abstractions.
* Simplicity, adding as few layers and new abstractions as possible.
* Ready to use, providing a full implementation of the API and RAG
  pipeline.

Main building blocks:
* APIs are defined in `private_gpt:server:<api>`. Each package contains an
  `<api>_router.py` (FastAPI layer) and an `<api>_service.py` (the
  service implementation). Each *Service* uses LlamaIndex base abstractions instead
  of specific implementations,
  decoupling the actual implementation from its usage.
* Components are placed in
  `private_gpt:components:<component>`. Each *Component* is in charge of providing
  actual implementations to the base abstractions used in the Services - for example
  `LLMComponent` is in charge of providing an actual implementation of an `LLM`
  (for example `LlamaCPP` or `OpenAI`).

## 💡 Contributing
Contributions are welcomed! To ensure code quality we have enabled several format and
typing checks, just run `make check` before committing to make sure your code is ok.
Remember to test your code! You'll find a tests folder with helpers, and you can run
tests using `make test` command.

Interested in contributing to PrivateGPT? We have the following challenges ahead of us in case
you want to give a hand:

### Improvements
- Better RAG pipeline implementation (improvements to both indexing and querying stages)
- Code documentation
- Expose execution parameters such as top_p, temperature, max_tokens... in Completions and Chat Completions
- Expose chunk size in Ingest API
- Implement Update and Delete document in Ingest API
- Add information about tokens consumption in each response
- Add to Completion APIs (chat and completion) the context docs used to answer the question
- In “model” field return the actual LLM or Embeddings model name used

### Features
- Implement concurrency lock to avoid errors when there are several calls to the local LlamaCPP model
- API key-based request control to the API
- CORS support
- Support for Sagemaker
- Support Function calling
- Add md5 to check files already ingested
- Select a document to query in the UI
- Better observability of the RAG pipeline

### Project Infrastructure
- Create a “wipe” shortcut in `make` to remove all contents of local_data folder except .gitignore
- Packaged version as a local desktop app (windows executable, mac app, linux app)
- Dockerize the application for platforms outside linux (Docker Desktop for Mac and Windows)
- Document how to deploy to AWS, GCP and Azure.

##

## 💬 Community
Join the conversation around PrivateGPT on our:
- [Twitter (aka X)](https://twitter.com/PrivateGPT_AI)
- [Discord](https://discord.gg/bK6mRVpErU)

## 📖 Citation
If you use PrivateGPT in a paper, check out the [Citation file](CITATION.cff) for the correct citation.  
You can also use the "Cite this repository" button in this repo to get the citation in different formats.

Here are a couple of examples:

#### BibTeX
```bibtex
@software{Martinez_Toro_PrivateGPT_2023,
author = {Martínez Toro, Iván and Gallego Vico, Daniel and Orgaz, Pablo},
license = {Apache-2.0},
month = may,
title = {{PrivateGPT}},
url = {https://github.com/imartinez/privateGPT},
year = {2023}
}
```

#### APA
```
Martínez Toro, I., Gallego Vico, D., & Orgaz, P. (2023). PrivateGPT [Computer software]. https://github.com/imartinez/privateGPT
```



################################################  Installation on Debian 11 Command line fallow the instructions  ########################


To install and run the `privateGPT` project on Debian 11, you can follow these step-by-step instructions:

1. **Install Dependencies:**

   First, ensure you have some basic dependencies installed on your system. Open a terminal and run the following commands:

   ```bash
   sudo apt update
   sudo apt install -y git make build-essential zlib1g-dev libssl-dev libbz2-dev \
                    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
                    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
                    libxml2-dev libxmlsec1-dev libffi-dev libcairo2-dev pkg-config
   ```

   These packages are required for building Python, managing dependencies, and running the project.

2. **Clone the Repository:**

   Navigate to the directory where you want to clone the `privateGPT` repository and execute the following commands:

   ```bash
   git clone https://github.com/imartinez/privateGPT
   cd privateGPT
   ```

3. **Install Python 3.11:**

   To install Python 3.11 using `pyenv`, follow these steps:

   ```bash
   # Install pyenv
   curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

   # Update your shell profile (e.g., .bashrc or .zshrc)
   echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
   echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
   source ~/.bashrc

   # Install Python 3.11
   pyenv install 3.11
   pyenv local 3.11
   ```

4. **Install Poetry:**

   Install `poetry`, a Python package manager and dependency manager, using `pip`:

   ```bash
   pip install poetry
   ```

5. **Install Additional Dependencies:**

   Install the project's dependencies with the following command:

   ```bash
   poetry install --with ui,local
   ```

6. **Download Embedding and LLM Models:**

   Download the necessary models by running:

   ```bash
   poetry run python scripts/setup
   ```

7. **Install llama-cpp-python (Optional):**

   If you're using a Mac with a Metal GPU, you can enable GPU support with `llama-cpp-python`. Run this command:

   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
   ```

8. **Run the Local Server:**

   Start the local server by running the following command:

   ```bash
   PGPT_PROFILES=local make run
   ```

9. **Access the UI:**

   Open a web browser and navigate to the following URL to access the `privateGPT` user interface:

   ```
   http://localhost:8001/
   ```

   You should now be able to try out the application.

That's it! You have successfully installed and run the `privateGPT` project on Debian 11. Make sure to follow any additional configuration or setup instructions specific to your use case, especially if you are working with GPU support on a platform other than macOS.




It appears that you don't have `pyenv` installed on your system. To install `pyenv`, you can follow these steps:

1. **Install `pyenv`:**

   Open a terminal and run the following commands to install `pyenv`:

   ```bash
   curl https://pyenv.run | bash
   ```

   This command will download and install `pyenv`.

2. **Update Your Shell Configuration:**

   After installing `pyenv`, you need to update your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`) to include `pyenv` in your path. Here's how you can do it for `bash`:

   ```bash
   echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
   echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
   source ~/.bashrc
   ```

   If you're using `zsh`, replace `~/.bashrc` with `~/.zshrc` in the above commands.

3. **Verify `pyenv` Installation:**

   To verify that `pyenv` is installed correctly, run:

   ```bash
   pyenv --version
   ```

   You should see the `pyenv` version information.

4. **Install Python 3.11:**

   Now that you have `pyenv` installed, you can proceed to install Python 3.11 as mentioned in the previous instructions:

   ```bash
   pyenv install 3.11
   ```

5. **Set Python 3.11 as the Local Version:**

   Set Python 3.11 as the local version for your project:

   ```bash
   pyenv local 3.11
   ```

6. **Verify the Python Version:**

   Verify that Python 3.11 is now the active version for your project by running:

   ```bash
   python --version
   ```

   It should display Python 3.11.

7. **Reinstall Dependencies:**

   After switching to Python 3.11, you may need to reinstall the project's dependencies. Run the following command again:

   ```bash
   poetry install --with ui,local
   ```

8. **Run the Project:**

   Finally, try running the project again:

   ```bash
   PGPT_PROFILES=local make run
   ```

These steps should help you set up `pyenv` and resolve the compatibility issue with Python versions for the `privateGPT` project on your Debian 11 system.
