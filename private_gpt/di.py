from injector import Injector
from llama_index import ServiceContext
from llama_index.llms.base import LLM
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt,
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import VectorStore

from private_gpt.constants import MODELS_PATH
from private_gpt.settings import settings


def create_application_injector() -> Injector:
    match settings.llm.mode:
        case "local":
            import chromadb
            from llama_index.llms import LlamaCPP

            local_settings = settings.local
            # LLM
            llm = LlamaCPP(
                model_path=str(MODELS_PATH / local_settings.model_name),
                temperature=0.1,
                # llama2 has a context window of 4096 tokens,
                # but we set it lower to allow for some wiggle room
                context_window=3900,
                generate_kwargs={},
                # set to at least 1 to use GPU
                model_kwargs={"n_gpu_layers": 1},
                # transform inputs into Llama2 format
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
            service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
            # VectorStore
            db = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = db.get_or_create_collection(
                "make_this_parametrizable_per_api_call"
            )  # TODO
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        case "sagemaker":
            from private_gpt.llm.sagemaker import SagemakerLLM

            sagemaker_settings = settings.sagemaker

            llm = SagemakerLLM(
                endpoint_name=sagemaker_settings.endpoint_name,
            )
            service_context = None  # TODO
            vector_store = None  # TODO

        case "openai":
            from llama_index.llms import OpenAI

            openai_settings = settings.openai.api_key
            llm = OpenAI(api_key=openai_settings)
            service_context = None  # TODO
            vector_store = None  # TODO

    injector = Injector(auto_bind=True)
    injector.binder.bind(LLM, llm)
    injector.binder.bind(ServiceContext, service_context)
    injector.binder.bind(VectorStore, vector_store)
    return injector


root_injector: Injector = create_application_injector()
