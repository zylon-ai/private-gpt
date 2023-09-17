# private-gpt

### Installation

Install `llama-cpp-python` on your venv

For OSX, using Metal, run the following

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
```

Install the rest of the requirements

```bash
pip installl -r requirements.txt
```

### Download the LLMs

So far, only llama-2-7b is configured, all models should be saved to the `models` folder.

It's a big download, it will take some time.

```bash
sh scripts/setup
```
