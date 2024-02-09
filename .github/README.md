# EC's Fork of 🔒 PrivateGPT 📑

This repo is a fork of [PrivateGPT](https://github.com/imartinez/privateGPT) whose APIs are utilized by the Immigration Commons Chatbot to provide real-time querying of documents.

This README holds important notes, instructions, and general information about our custom implementation of PrivateGPT. [Click here to read the original PrivateGPT README](../README.md). As per PrivateGPT's maintainers, the most up-to-date documentation is found on the official [PrivateGPT website](https://docs.privategpt.dev/)

## Why PrivateGPT and what parts of the open source project do we use?

PrivateGPT is an open source AI-chatbot project that allows users to upload and ask questions about their documents. It is a production-ready and modular project that lends itself to custom implementations of the high- and low-level APIs, services, and component layers. It also has a frontend component but we make _no_ use of it since it's built on Gradio UI and, unfortunately, does not meet ImmigrationCommons' UI needs and requirements.

As of February 1, 2024 we decided to use PrivateGPT's:

- High-level APIs - they abstract all the complexity of a RAG (Retrieval Augmented Generation) pipeline implementation _and thus_ should help us more quickly build the first iteration of the `Chatbot` UI in ImmigrationCommons.
- `local` settings - these are the default settings which make it easy to experiment and test locally. It provides Qdrant as the default vector store and Mistral-7B as the LLM.

## Installation and Setup Instructions

There are two ways to run PrivateGPT. We strongly recommend the first option:

1. **Docker (recommended):**

   - To make sure you build and run PrivateGPT using `docker-compose`, please run the following commands in the order provided:
     a. Build image: `docker compose build`
     b. Download local models: `docker compose run --rm --entrypoint="bash -c '[ -f scripts/setup ] && scripts/setup'" private-gpt`
     c. Run the service: `docker-compose up`

2. Local (will most likely require installation of llama.cpp among other custom configurations):
   - Go to [PrivateGPT's installation page](https://docs.privategpt.dev/installation) and follow the instructions there.
   - GOTCHA: if you have an M1/M2 machine and run into `llama.cpp`-related errors after running `poetry run python scripts/setup`, go to the "OSX GPU support" section on the same page and follow the instructions listed there, THEN,
     - re-run: `poetry install --with local` and lastly: `poetry run python -m private_gpt`

## Branching Strategy, Committing Changes, Merging and Maintenance

It's important to reiterate that we are using a fork of PrivateGPT and as a result we have separate branches for the `staging` and `production` environments. Below are a list of the most important branches and how they should be utilized:

- `main` - **_NEVER_** merge any custom code into our fork's `main` branch - we will use it to pull the latest changes from the original [PrivateGPT project](https://github.com/imartinez/privateGPT) and eventually merge them into `ec-main`.
- `ec-main` - This is our `production` (aka `main`) branch.
- `ec-dev` - This is our `staging` environment's branch. We chose this name in order to maintain some naming parity with Immigration Commons's `dev` branch which is used by the `staging` environment.
- `IV-##-my-feature` - Recommended naming structure of feature branches - the prefix `IV-##` corresponds to the JIRA ticket's label and it's necessary for tracking its progress in JIRA.

### Maintenance

We should aim to pull and make use of the latest changes from the original [PrivateGPT repo](https://github.com/imartinez/privateGPT) as often as possible. Depending on how much we have customized PrivateGPT, this might require some serious effort and work into fixing code-breaking changes, resolving merge conflicts, etc.

Please follow these (suggested) steps when pulling the latest changes from the original PrivateGPT:

- Go to our [forked repo's page](https://github.com/emersoncollective/ec-private-gpt)
- Click on the `Sync fork` button (this should bring our fork's `main` branch up-to-date)
- Branch out from `ec-dev` - ex: `git checkout -b feature-sync-ec-fork`
- Merge `main` into the `feature-sync-ec-fork` branch
- Resolve merge conflicts and fix code-breaking changes.
- Push branch `git push origin feature-sync-ec-fork`
- Submit a Pull Request as usual to bring the `ec-main` and `ec-dev` branches up-to-date.
