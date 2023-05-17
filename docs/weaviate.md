# Weaviate-specific Setup

1. Deploy weaviate using docker compose
   Here's a bare minimum docker compose file:
   ```bash
   curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.19.3"
   ```

2. Install Langchain's master branch (do this until PR [#4824](https://github.com/hwchase17/langchain/pull/4824) is released):
   ```bash
   pip install git+https://github.com/hwchase17/langchain.git@main
   ```

3. Define the following env vars:
   ```
   VECTOR_STORE=weaviate
   WEAVIATE_URL=http://localhost:8080
   ```