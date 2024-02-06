# EC's Forked (Submodule) Version of 🔒 PrivateGPT 📑

This [forked version](https://github.com/emersoncollective/ec-private-gpt) of [PrivateGPT](https://github.com/imartinez/privateGPT) has been added as a Git submodule of Immigration Commons.

This README holds important notes, instructions, and general information about our custom implementation of PrivateGPT. [Click here to read the original PrivateGPT README](../README.md). As per PrivateGPT's maintainers, the most up-to-date documentation is found on the official [PrivateGPT website](https://docs.privategpt.dev/)

## Why PrivateGPT and what parts of the open source project do we use?

PrivateGPT is an open source AI-chatbot project that allows users to upload and ask questions about their documents. It is a production-ready and modular project that lends itself to custom implementations of the high- and low-level APIs, services, and components layers. It also has a frontend component but we make _no_ use of it since it's built on Gradio UI and, unfortunately, does not meet ImmigrationCommons' UI needs and requirements.

As of February 1, 2024 we decided to use PrivateGPT's:

- High-level APIs - they abstract all the complexity of a RAG (Retrieval Augmented Generation) pipeline implementation to _and thus_ will help us speed up the development of the `Chatbot` UI in ImmigrationCommons.
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

## Branching Strategy, Commiting Changes, Merging and Maintenance

It's important to reiterate that we are using a forked version of PrivateGPT which we added as a `submodule` to our ImmigrationCommons repo. While this configuration allows us to have a cleaner separation of the ImmigrationCommons and `ec-private-gpt` repositories and easily retrieve submodule-level updates, it does come with some drawbacks.

### Branching Strategy:

To make sure we are able to utilize and have an easier time merging the latest updates from the original PrivateGPT repository, the following branch rules should be followed:

- `main` - **_NEVER_** merge any custom code into our forked PrivateGPT `main` branch - we will use it to pull the latest changes from the original [PrivateGPT project](https://github.com/imartinez/privateGPT).
- `ec-main` - Think of this as our `main` branch
- `ec-dev` - Our dev branch. We should treat it the same way we treat the `dev` branch on ImmigrationCommons - that is, as the `staging` environment's base branch.
- `IV-##-my-feature` - Suggested name structure of feature branches. Remember, when you create a feature branch inside the `ec-private-gpt` submodule, the branch is a branch of our forked `ec-private-gpt` repo **and not** the Immigration Commons repo.

### Building, Committing, and Merging New Features

Let's assume you want to build a new feature and make it available in ImmigrationCommons; you'd want to follow these instructions:

- `git submodule update --recursive --remote` - just like running `git pull`
- `cd ec-private-gpt`
- `git checkout -b IV-001-my-cool-feature`
- Make your changes
- `git add .`
- `git commit -m "Implement cool feature in ec-private-gpt submodule."`
- `git push origin IV-001-my-cool-feature`
- create and submit Pull Request to merge the feature branch into the `ec-staging` branch of our forked `ec-private-gpt` repo.
- once the PR has been approved, `cd` one level up to the root of ImmigrationCommons.
- create a new branch to ensure we can utilize the latest feature code in our `ec-private-gpt` submodule: `git checkout -b feature-update-submodule`, then:
- `git submodule update --recursive --remote` - updates submodule to the latest commits
- `git merge --no-ff ec-staging` - merge the submodule changes in `ec-staging`
- `git push origin feature-update-submodule`
- then, submit a PR on ImmigrationCommons as usual

Please note, that once changes have been approved on `staging`, the same exact steps will be required to merge the latest updates in `staging` into `production`.

### Maintenance

We should aim to pull and make use of the latest changes from the original [PrivateGPT repo](https://github.com/imartinez/privateGPT) as often as possible. Depending on how much we have customized PrivateGPT, this might require some serious effort and work into fixing code-breaking changes, resolving merge conflicts, etc.

Please follow these (suggested) steps when pulling the latest changes from the original PrivateGPT:

- Go to our [forked repo's page](https://github.com/emersoncollective/ec-private-gpt)
- Click on the `Sync fork` button (this should bring the forked repo's `main` branch up-to-date)
- On VS Code and on the terminal, run: `git submodule update --recursive --remote`
- Branch out from `ec-staging` - ex: `git checkout -b feature-sync-ec-fork`
- Resolve merge conflicts and fix code-breaking changes.
- Push branch `git push origin feature-sync-ec-fork`
- Submit a Pull Request as usual. Refer to the [Building, Committing, and Merging New Features](Building) section for more detailed instructions on `submodule` PRs.
