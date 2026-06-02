"""private-gpt."""

import logging
import os
import warnings

from pydantic import PydanticDeprecatedSince20

# Set to 'DEBUG' to have extensive logging turned on, even for libraries
ROOT_LOG_LEVEL = "INFO"

PRETTY_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-8s] [%(threadName)s] %(name)+25s - %(message)s"
logging.basicConfig(level=ROOT_LOG_LEVEL, format=PRETTY_LOG_FORMAT, datefmt="%H:%M:%S")
logging.captureWarnings(True)


# adding tiktoken cache path within repo to be able to run in offline environment.
os.environ["TIKTOKEN_CACHE_DIR"] = "tiktoken_cache"

# Disable warning tokenizer about torch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable deprecation of opentelemetry. It will raise at the end of 2025
# The pkg_resources package is slated for removal as early as 2025-11-30.
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

# Disable distutils deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

# Disable deprecation warnings from pydantic about the v2 changes, which we are not yet
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

# Disable DeepEval update warning
os.environ["DEEPEVAL_UPDATE_WARNING_OPT_OUT"] = "NO"

# disable logs from llama_index.core.indices.loading
logging.getLogger("llama_index.core.indices.loading").setLevel(logging.WARNING)

# Disable oras logs
logging.getLogger("oras.logger").setLevel(logging.ERROR)

# Disable httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
