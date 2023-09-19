# private-gpt

### Installation

Install `llama-cpp-python` on your venv

For OSX, using Metal, run the following

```bash
poetry remove llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" poetry add llama-cpp-python
```

Install (or update) the rest of the requirements

```bash
poetry install
```

### Download the LLMs

So far, only llama-2-7b is configured, all models should be saved to the `models` folder.

It's a big download, it will take some time.

```bash
sh scripts/setup
```
