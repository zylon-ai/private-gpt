# Documentation of PrivateGPT

The documentation of this project is being rendered thanks to [fern](https://github.com/fern-api/fern).

Fern is basically transforming your `.md` and `.mdx` files into a static website: your documentation.

The configuration of your documentation is done in the `./docs.yml` file.
There, you can configure the navbar, tabs, sections and pages being rendered.

The documentation of fern (and the syntax of its configuration `docs.yml`) is 
available there [docs.buildwithfern.com](https://docs.buildwithfern.com/).

## How to run fern

**You cannot render your documentation locally without fern credentials.**

To see how your documentation looks like, you **have to** use the CICD of this
repository (by opening a PR, CICD job will be executed, and a preview of 
your PR's documentation will be deployed in vercel automatically, through fern).

The only thing you can do locally, is to run `fern check`, which check the syntax of
your `docs.yml` file.

## How to add a new page
Add in the `docs.yml` a new `page`, with the following syntax:

```yml
navigation:
  # ...
  - tab: my-existing-tab
    layout:
      # ...
      - section: My Existing Section
        contents:
          # ...
          - page: My new page display name
            # The path of the page, relative to `fern/`
            path: ./docs/pages/my-existing-tab/new-page-content.mdx
```