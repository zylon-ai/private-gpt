# Fern Docs

This folder contains the Fern configuration and the checked-in documentation pages for the root repository.

Most page content mirrors the newer docs content from the sibling `private-gpt` checkout, but the published site for this repo is driven from `fern/docs.yml` and `fern/openapi/openapi.json`.

## Common tasks

- Refresh the OpenAPI spec: run `make api-docs` from the repo root.
- Validate the Fern config locally: run `cd fern && fern check`.
- Update navigation: edit `fern/docs.yml`.

## Adding a page

Add the page file under `fern/docs/pages/...`, then register it in `fern/docs.yml`:

```yml
navigation:
  - tab: manual
    layout:
      - section: My Section
        contents:
          - page: My page
            path: ./docs/pages/manual/my-page.mdx
```
