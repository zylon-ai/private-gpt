# private-gpt

[![Tests](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml)

### Installation

This project requires poetry: https://python-poetry.org/docs/

Install the dependencies with.

```bash
poetry install --with ui,local
```

The `ui` and `local` are optional groups if you need that functionality.

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

#### Environment variables expansion

Configuration files can contain environment variables,
they will be expanded at runtime.

Expansion must follow the pattern `${VARIABLE_NAME:default_value}`.

For example, the following configuration will use the value of the `PORT`
environment variable or `8001` if it's not set.

```yaml
server:
  port: ${PORT:8001}
```

Missing variables with no default will produce an error.

### Running with a local LLM

For LLM to be usable a GPU acceleration is required
(CPU execution is possible, but extremely slow), however,
typical Macbook laptops or window desktops lack GPU memory to run
even the smallest LLMs. For that reason
**local execution is only supported for models compatible
with [llama.cpp](https://github.com/ggerganov/llama.cpp)**

In particular, these two models (with their quantized variants)
work particularly well:

* https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF
* https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

Select the quantized version of the model that fits your GPU,
download it and place it under `models/your_quantized_model_of_choice.gguf`.

In your `settings-local.yaml` configure the model to use it:

```yaml
llm:
  default_llm: local # Use the local model by default

local:
  enabled: true
  model_name: mistral-7b-instruct-v0.1.Q4_0.gguf # The name of the model you downloaded
```