# Design Plan for Vector Database Integration

## 1. Vector Database Choice

For simplicity and local execution, we will use `ChromaDB` as the vector store. It's lightweight and easy to integrate.

## 2. Embedding Model

We will use `SentenceTransformers` with a pre-trained model like `all-MiniLM-L6-v2` for generating embeddings. This model is efficient and provides good general-purpose embeddings.

## 3. Modifications to `SimpleKnowledgeBase`

### a. Initialization (`__init__` and `init_db`)

- Remove `documents_fts` table creation.
- Initialize `ChromaDB` client and collection.

### b. Adding Documents (`add_document`)

- Generate embeddings for the document content using the chosen embedding model.
- Store the document content, metadata, and its embedding in the ChromaDB collection.
- The existing SQLite `documents` table will still be used to store the original document content and metadata, and ChromaDB will store the embeddings and a reference to the document ID.

### c. Searching Documents (`search_documents`)

- Generate an embedding for the search query.
- Perform a similarity search against the ChromaDB collection using the query embedding.
- Retrieve the relevant document IDs from ChromaDB and then fetch the full document details from the SQLite `documents` table.

## 4. Dependencies

Add `chromadb` and `sentence-transformers` to `requirements.txt`.

## 5. Configuration

Update `knowledge_base_config.yaml` to include paths for ChromaDB storage and potentially the embedding model name.
