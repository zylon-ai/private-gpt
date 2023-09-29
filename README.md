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

### Settings

PrivateGPT can be configured through env variables
using yaml files and profiles.

#### `PGPT_SETTINGS_FOLDER`

The location of the settings folder. Defaults to the root of the project.

#### `PGPT_PROFILES`

Additional profiles to load, format is a comma separated list of profile names.
This will merge `settings-{profile}.yaml` on top of the base settings file.

For example:
`PGPT_PROFILES=local,cuda` will load `settings-local.yaml` and `settings-cuda.yaml`,
their contents will be merged with later profiles properties overriding earlier ones.

During testing, the `test` profile will be active along with the default.

#### Environment variables expansion

Configuration files can contain environment variables, they will be expanded at runtime,

They have to follow the pattern `${VARIABLE_NAME:default_value}`.

For example, the following configuration will use the value of the `PORT`
environment variable or `8001` if it's not set.

```yaml
server:
  port: ${PORT:8001}
```

Missing variables with no default will produce an error.