# Weaviate-specific Setup

1. Deploy weaviate using docker compose
   Here's a bare minimum docker compose file:
   ```bash
   curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.19.3"
   ```

2. Install Langchain's PR #4365:
   ```bash
   pip install git+https://github.com/hwchase17/langchain.git@refs/pull/4365/head
   ```

3. Define the following env vars:
   ```
   VECTOR_STORE=weaviate
   WEAVIATE_URL=http://localhost:8080
   ```