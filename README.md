# Que Agents

## Project Overview

Que Agents is a project designed to provide intelligent agents for various tasks, including customer support and marketing. It leverages a knowledge base to provide relevant information and automate responses. This project has been recently updated to utilize a vector database for enhanced knowledge retrieval.

## Features

- **Intelligent Agents**: Specialized agents for customer support and marketing.
- **Knowledge Base**: A robust knowledge base for storing and retrieving information.
- **Vector Database Integration**: Replaced Full Text Search with a vector database (ChromaDB) for more accurate and semantic search capabilities.
- **Embedding Model**: Uses `all-MiniLM-L6-v2` for generating document and query embeddings.
- **Data Loading**: Supports loading data from Markdown, JSON, CSV, and PostgreSQL sources.

## Setup and Installation

To set up the project, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/Behordeun/que_agents.git
cd que_agents
```

1. **Install dependencies**:

```bash
make install
   ```

This will install all required Python packages listed in `requirements.txt`.

1. **Configure the knowledge base**:

Ensure your `configs/knowledge_base_config.yaml` file is configured correctly. It should look like this:

```yaml
knowledge_base:
  db_path: knowledge_base.db
  chroma_path: ./chroma_db
  embedding_model: all-MiniLM-L6-v2
```

Also, rename `configs/database_config_example.yaml` to `configs/database_config.yaml` and configure your PostgreSQL database if you plan to use it.

## Usage

### Running the Application

To start the FastAPI application, use the following command:

```bash
make run
```

This will typically start the Uvicorn server, and you can access the API at `http://127.0.0.1:8000` (or the port specified in your configuration).

### Testing

To run the tests, use:

```bash
make test
```

### Cleaning Up

To remove generated files (like `.pyc` files, `__pycache__` directories, SQLite database, and ChromaDB data), use:

```bash
make clean
```

## Project Structure

```plain text
que_agents/
├── configs/
│   ├── agent_config.yaml
│   ├── api_config.yaml
│   ├── database_config.yaml
│   └── knowledge_base_config.yaml
├── data/
│   ├── semi_structured/
│   ├── structured/
│   └── unstructured/
├── docs/
├── src/
│   └── que_agents/
│       ├── agents/
│       ├── api/
│       ├── core/
│       │   └── database.py
│       ├── knowledge_base/
│       │   └── kb_manager.py
│       ├── template/
│       └── utils/
├── tests/
├── .gitignore
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Vector Database Implementation Details

- **ChromaDB**: Used as the primary vector store for efficient similarity search.
- **Sentence Transformers**: The `all-MiniLM-L6-v2` model is used to generate embeddings for both documents and search queries.
- **`kb_manager.py`**: This file has been updated to:

- Initialize ChromaDB client and collection.
- Generate and store embeddings in ChromaDB when adding new documents.
- Perform vector similarity search using query embeddings and retrieve relevant documents from ChromaDB, then fetch full details from the SQLite database.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
