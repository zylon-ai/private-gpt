# PrivateGPT

PrivateGPT is a production-ready AI project that allows you to ask questions to your documents using the power of Large Language Models (LLMs), even in scenarios without Internet connection. 
100% private, no data leaves your execution environment at any point.

The project provides an API offering all the primitives required to build private, context-aware AI applications. It follows and extends [OpenAI API standard](https://openai.com/blog/openai-api), and supports both normal and streaming responses.

The API is divided into two logical blocks:

**High-level API**, which abstracts all the complexity of a RAG (Retrieval Augmented Generation) pipeline implementation:
- Ingestion of documents: internally managing document parsing, splitting, metadata extraction, embedding generation and storage.
- Chat & Completions using context from ingested documents: abstracting the retrieval of context, the prompt engineering and the response generation.

**Low-level API**, which allows advanced users to implement their own complex pipelines:
- Embeddings generation: based on a piece of text.
- Contextual chunks retrieval: given a query, returns the most relevant chunks of text from the ingested documents.

In addition to this, a working [Gradio UI](https://www.gradio.app/) client is provided to test the API, together with a set of useful tools such as bulk model download script, ingestion script, documents folder watch, etc.

<img width="900"  alt="demo" src="https://lh3.googleusercontent.com/drive-viewer/AK7aPaDEVquu0igr1Bgkq8dCR2z4RDMXDd9JTbIV1LtZhxadIm-jmoTZnku3UrRMf4OWx-o_MkbpmOmcu7TJ9f3tNrAfSUumvQ=s1600">

> 👂 **Need help applying PrivateGPT to your specific use case?** [Let us know more about it](https://forms.gle/4cSDmH13RZBHV9at7) and we'll try to help! We are refining PrivateGPT through your feedback.

## 🎞️ Overview
DISCLAIMER: This README is not updated as frequently as the [documentation](https://docs.privategpt.dev/). Please check it out for the latest updates!

### Motivation behind PrivateGPT
Generative AI is a game changer for our society, but adoption in companies of all size and data-sensitive domains like healthcare or legal is limited by a clear concern: **privacy**. 
Not being able to ensure that your data is fully under your control when using third-party AI tools is a risks these industries cannot allow.

### Primordial version
The first version of PrivateGPT was launched in May 2023 as a novel approach to address the privacy concern by using LLMs in a complete offline way. 
This was done by leveraging existing technologies developed by the thriving Open Source AI community: [LangChain](https://github.com/hwchase17/langchain), [LlamaIndex](https://www.llamaindex.ai/), [GPT4All](https://github.com/nomic-ai/gpt4all), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/).

That version, which rapidly became a go-to project for privacy-sensitive setups and served as the seed for thousands of local-focused generative AI projects, was the foundation of what PrivateGPT is becoming nowadays; 
thus a simpler and more educational implementation to understand the basic concepts required to build a fully local -and therefore, private- chatGPT-like tool.

If you want to keep experimenting with it, we have saved it in the [primordial branch](TBD) of the project.

### Present and Future of PrivateGPT
PrivateGPT is now evolving towards becoming a gateway to generative AI models and primitives, including completions, document ingestion, RAG pipelines and other low-level building blocks. 
We want to make easier for any developer to build AI applications and experiences, as well as providing a suitable extensive architecture for the community to keep contributing.   

Stay tuned to our [releases](TBD) to check all the new features and changes included.

## 📄 Documentation
Full documentation on installation, dependencies, configuration, running the server, deployment options, ingesting local documents and API details can be found here: https://docs.privategpt.dev/

This documentation is automatically generated every time the server is launched. You can find the script in charge of doing so in `scripts/extract_openapi.py`

## Architecture
TBD

## 💡 Contributing
Interested in contributing to PrivateGPT? See our Contribution Guide for more details -> TBD

### TODOs
TBD

## 📖 Citation
Reference to cite if you use PrivateGPT in a paper:

```
@software{PrivateGPT_2023,
authors = {Martinez, I., Gallego, D. Orgaz, P.},
month = {5},
title = {PrivateGPT},
url = {https://github.com/imartinez/privateGPT},
year = {2023}
}
```