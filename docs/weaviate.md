# Weaviate-specific Setup

1. Download the Docker compose file:
   ```bash
   curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.19.3"
   ```

2. Deploy Weaviate using Docker:
   ```bash
   docker compose up -d
   ```
   
3. Install Langchain's master branch (do this until PR [#4824](https://github.com/hwchase17/langchain/pull/4824) is released):
   ```bash
   pip install git+https://github.com/hwchase17/langchain.git@master
   ```

4. Define these additional env vars in `.env`:
   ```
   VECTOR_STORE=weaviate
   WEAVIATE_URL=http://localhost:8080
   ```
