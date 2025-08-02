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

### Clone the repository

```bash
git clone https://github.com/Behordeun/que_agents.git
cd que_agents
```

### Install dependencies

```bash
make install
```

This will install all required Python packages listed in `requirements.txt`.

### Configure the knowledge base

Ensure your `configs/knowledge_base_config.yaml` file is configured correctly. It should look like this:

```yaml
knowledge_base:
  db_path: knowledge_base.db
  chroma_path: ./chroma_db
  embedding_model: all-MiniLM-L6-v2
```

Also, rename `configs/database_config_example.yaml` to `configs/database_config.yaml` and configure your PostgreSQL database if you plan to use it.

### Configure LLM (Large Language Model) platforms

The project supports multiple LLM providers (OpenAI, Groq, Anthropic, Azure OpenAI, Local Ollama). You can configure this in `configs/llm_config.yaml`.

```yaml
llm:
  default_provider: "openai" # Change to "groq", "anthropic", "azure_openai", or "local"
  temperature:
    customer_support: 0.3
    marketing: 0.7
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      # ... other OpenAI settings
    groq:
      api_key: "${GROQ_API_KEY}"
      # ... other Groq settings
    # ... other providers
```

Ensure you set the appropriate API keys as environment variables (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`).

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
├── api_endpoints.json
├── configs
│   ├── agent_config.yaml
│   ├── api_config.yaml
│   ├── database_config_example.yaml
│   ├── database_config.yaml
│   ├── knowledge_base_config.yaml
│   ├── llm_config_example.yaml
│   └── llm_config.yaml
├── data
│   ├── semi_structured
│   │   ├── campaign_performance.csv
│   │   └── customer_feedback.csv
│   ├── structured
│   │   └── sample_responses.json
│   └── unstructured
│       ├── company_policies.md
│       ├── faq_database.md
│       └── product_documentation.md
├── docs
│   ├── Customer Support & Marketing Agents.md
│   ├── framework_research.md
│   ├── Project Summary.md
│   ├── Technical Documentation.md
│   └── vector_db_design_plan.md
├── integration_test_report.json
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   └── que_agents
│       ├── __init__.py
│       ├── agents
│       │   ├── __init__.py
│       │   ├── customer_support_agent.py
│       │   └── marketing_agent.py
│       ├── api
│       │   ├── __init__.py
│       │   └── main.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── llm_factory.py
│       ├── knowledge_base
│       │   ├── __init__.py
│       │   └── kb_manager.py
│       └── utils
│           ├── __init__.py
│           └── data_populator.py
├── template
│   └── index.html
└── tests
    ├── __init__.py
    └── integration_test.py
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
