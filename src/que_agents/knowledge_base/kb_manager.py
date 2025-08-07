# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module provides utilities for managing the knowledge base

import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from src.que_agents.core.database import KnowledgeBase, get_session

# Load knowledge base configuration
with open("configs/knowledge_base_config.yaml", "r") as f:
    kb_config = yaml.safe_load(f)

# Load LLM configuration
with open("configs/llm_config.yaml", "r") as f:
    llm_config = yaml.safe_load(f)


class SimpleKnowledgeBase:
    """Simple knowledge base implementation using SQLite for vector storage simulation"""

    def __init__(self):
        self.db_path = kb_config["knowledge_base"]["db_path"]
        self.chroma_path = kb_config["knowledge_base"]["chroma_path"]
        self.embedding_model_name = kb_config["knowledge_base"]["embedding_model"]
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.init_db()

    def init_db(self):
        """Initialize the knowledge base database and ChromaDB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_path TEXT,
                category TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="que_agents_kb"
        )

    def add_document(
        self,
        title: str,
        content: str,
        source_type: str,
        source_path: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Add a document to the knowledge base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute(
            """
            INSERT INTO documents (title, content, source_type, source_path, category, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (title, content, source_type, source_path, category, metadata_json),
        )

        doc_id = cursor.lastrowid

        # Generate embedding and add to ChromaDB
        document_text = f"{title}. {content}"
        embedding = self.embedding_model.encode(document_text).tolist()
        self.chroma_collection.add(
            embeddings=[embedding],
            documents=[document_text],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source_type": source_type,
                    "category": category or "",
                }
            ],
            ids=[str(doc_id)],
        )

        conn.commit()
        conn.close()

        return doc_id

    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents using vector similarity search with ChromaDB"""
        try:
            if not query or not query.strip():
                return []

            query_embedding = self.embedding_model.encode(query).tolist()

            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["metadatas"],
            )

            # Check if results exist and have metadatas
            if not results or not results.get("metadatas"):
                print(f"No results found for query: {query}")
                return []

            metadatas = results["metadatas"]

            # Check if metadatas is empty or contains empty lists
            if not metadatas or not metadatas[0]:
                print(f"Empty metadata for query: {query}")
                return []

            # Safely extract document IDs with error checking
            doc_ids = []
            try:
                for meta_list in metadatas:
                    if meta_list and len(meta_list) > 0:
                        # Check if the metadata has doc_id and it's not None
                        if (
                            isinstance(meta_list[0], dict)
                            and "doc_id" in meta_list[0]
                            and meta_list[0]["doc_id"] is not None
                        ):
                            doc_ids.append(int(meta_list[0]["doc_id"]))
                        else:
                            print(
                                f"Warning: Missing or None doc_id in metadata: {meta_list[0]}"
                            )
            except (ValueError, KeyError, TypeError, IndexError) as e:
                print(f"Error extracting doc_ids: {e}")
                return []

            if not doc_ids:
                print(f"No valid doc_ids found for query: {query}")
                return []

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                # Fetch full document details from SQLite
                placeholders = ",".join("?" * len(doc_ids))
                cursor.execute(
                    f"SELECT id, title, content, source_type, source_path, category, metadata, created_at FROM documents WHERE id IN ({placeholders})",
                    doc_ids,
                )

                sql_results = cursor.fetchall()
            except Exception as e:
                print(f"Error querying SQLite: {e}")
                return []
            finally:
                conn.close()

            if not sql_results:
                print(f"No documents found in SQLite for doc_ids: {doc_ids}")
                return []

            # Order results based on ChromaDB's ranking
            ordered_results = []
            for doc_id in doc_ids:
                for row in sql_results:
                    if row[0] == doc_id:
                        try:
                            metadata = json.loads(row[6]) if row[6] else {}
                        except json.JSONDecodeError:
                            metadata = {}

                        ordered_results.append(
                            {
                                "id": row[0],
                                "title": row[1],
                                "content": row[2],
                                "source_type": row[3],
                                "source_path": row[4],
                                "category": row[5],
                                "metadata": metadata,
                                "created_at": row[7],
                            }
                        )
                        break

            return ordered_results

        except Exception as e:
            print(f"Error in search_documents: {e}")
            return []

    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Get all documents in a specific category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, title, content, source_type, source_path, category, metadata, created_at
            FROM documents
            WHERE category = ?
            ORDER BY created_at DESC
        """,
            (category,),
        )

        results = []
        for row in cursor.fetchall():
            metadata = json.loads(row[6]) if row[6] else {}
            results.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "source_type": row[3],
                    "source_path": row[4],
                    "category": row[5],
                    "metadata": metadata,
                    "created_at": row[7],
                }
            )

        conn.close()
        return results


class AgentKnowledgeBase:
    """Agent-specific knowledge base functionality"""

    def __init__(self, agent_type: str, base_kb: SimpleKnowledgeBase):
        self.agent_type = agent_type
        self.base_kb = base_kb
        self.config = kb_config["knowledge_base"]["agents"].get(agent_type, {})

    def load_agent_data(self):
        """Load agent-specific data into knowledge base"""
        agent_data_dir = Path(f"data/{self.agent_type}")
        if not agent_data_dir.exists():
            print(f"Agent data directory not found: {agent_data_dir}")
            return 0

        loader = DocumentLoader(self.base_kb)
        total_loaded = 0

        # Load structured data
        structured_dir = agent_data_dir / "structured"
        if structured_dir.exists():
            files = loader.load_directory(
                str(structured_dir), [f"{self.agent_type}_structured"]
            )
            total_loaded += len(files)
            print(f"Loaded {len(files)} structured files for {self.agent_type}")

        # Load semi-structured data
        semi_structured_dir = agent_data_dir / "semi_structured"
        if semi_structured_dir.exists():
            files = loader.load_directory(
                str(semi_structured_dir), [f"{self.agent_type}_semi_structured"]
            )
            total_loaded += len(files)
            print(f"Loaded {len(files)} semi-structured files for {self.agent_type}")

        # Load unstructured data
        unstructured_dir = agent_data_dir / "unstructured"
        if unstructured_dir.exists():
            files = loader.load_directory(
                str(unstructured_dir), [f"{self.agent_type}_unstructured"]
            )
            total_loaded += len(files)
            print(f"Loaded {len(files)} unstructured files for {self.agent_type}")

        return total_loaded

    def search_agent_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Search knowledge base with agent-specific context"""
        # Try agent-specific categories first
        categories = self.config.get("categories", [])
        results = []

        for category in categories:
            category_results = self.base_kb.get_documents_by_category(
                f"{self.agent_type}_{category}"
            )
            query_lower = query.lower()
            for doc in category_results:
                if (
                    query_lower in doc["content"].lower()
                    or query_lower in doc["title"].lower()
                ):
                    results.append(doc)

        # If no agent-specific results, fall back to general search
        if not results:
            results = self.base_kb.search_documents(query, limit)

        return results[:limit]


def initialize_agent_knowledge_bases():
    """Initialize knowledge bases for all agents"""
    base_kb = SimpleKnowledgeBase()

    agents = [
        "customer_support",
        "marketing",
        "financial_trading_bot",
        "personal_virtual_assistant",
    ]
    agent_kbs = {}

    for agent_type in agents:
        agent_kb = AgentKnowledgeBase(agent_type, base_kb)
        loaded_count = agent_kb.load_agent_data()
        agent_kbs[agent_type] = agent_kb
        print(f"Initialized {agent_type} knowledge base with {loaded_count} documents")

    return base_kb, agent_kbs


def search_agent_knowledge_base(
    agent_type: str, query: str, limit: int = 5
) -> List[Dict]:
    """Search knowledge base for specific agent"""
    try:
        base_kb = SimpleKnowledgeBase()
        agent_kb = AgentKnowledgeBase(agent_type, base_kb)
        return agent_kb.search_agent_knowledge(query, limit)
    except Exception as e:
        print(f"Error searching {agent_type} knowledge base: {e}")
        return _get_agent_fallback_results(agent_type, query, limit)


def _get_agent_fallback_results(
    agent_type: str, query: str, _limit: int = 5
) -> List[Dict]:
    """Generate agent-specific fallback results"""
    if agent_type == "financial_trading_bot":
        return [
            {
                "id": f"fallback_{agent_type}_1",
                "title": "Market Analysis Best Practices",
                "content": f"Technical analysis guidelines for {query}. Consider RSI, MACD, moving averages, and market sentiment indicators for informed trading decisions.",
                "source_type": "fallback",
                "category": "trading",
                "metadata": {"agent": agent_type, "query": query},
            },
            {
                "id": f"fallback_{agent_type}_2",
                "title": "Risk Management Strategies",
                "content": f"Risk management protocols for {query}. Implement position sizing limits, stop-loss orders, and portfolio diversification strategies.",
                "source_type": "fallback",
                "category": "risk_management",
                "metadata": {"agent": agent_type, "query": query},
            },
        ]
    elif agent_type == "personal_virtual_assistant":
        return [
            {
                "id": f"fallback_{agent_type}_1",
                "title": "Personal Assistant Capabilities",
                "content": f"Personal assistance features for {query}. Available functions include smart home control, reminder management, weather information, and general queries.",
                "source_type": "fallback",
                "category": "personal_productivity",
                "metadata": {"agent": agent_type, "query": query},
            }
        ]

    return []


class DocumentLoader:
    """Load documents from various sources into the knowledge base"""

    def __init__(self, knowledge_base: SimpleKnowledgeBase):
        self.kb = knowledge_base

    def load_markdown_file(self, file_path: str, category: Optional[str] = None):
        """Load a markdown file into the knowledge base"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract title from first line if it's a header
        lines = content.split("\n")
        title = (
            lines[0].replace("#", "").strip()
            if lines and lines[0].startswith("#")
            else Path(file_path).stem
        )

        return self.kb.add_document(
            title=title,
            content=content,
            source_type="markdown",
            source_path=file_path,
            category=category,
            metadata={"file_size": os.path.getsize(file_path)},
        )

    def load_json_file(self, file_path: str, category: Optional[List[str]] = None):
        """Load a JSON file into the knowledge base"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert JSON to readable text
        content = json.dumps(data, indent=2)
        title = Path(file_path).stem

        # Ensure category is a string or None
        cat = (
            category[0]
            if isinstance(category, list) and category
            else category if isinstance(category, str) else None
        )
        return self.kb.add_document(
            title=title,
            content=content,
            source_type="json",
            source_path=file_path,
            category=cat,
            metadata={
                "file_size": os.path.getsize(file_path),
                "json_keys": list(data.keys()) if isinstance(data, dict) else [],
            },
        )

    def load_csv_file(self, file_path: str, category: Optional[str] = None):
        """Load a CSV file into the knowledge base"""
        content_lines = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            content_lines.append(f"CSV Headers: {', '.join(headers or [])}")
            content_lines.append("")

            # Add sample rows
            for i, row in enumerate(reader):
                if i < 10:  # Limit to first 10 rows for content
                    content_lines.append(f"Row {i+1}: {dict(row)}")
                else:
                    content_lines.append(
                        f"... and {sum(1 for _ in reader) + 1} more rows"
                    )
                    break

        content = "\n".join(content_lines)
        title = Path(file_path).stem

        return self.kb.add_document(
            title=title,
            content=content,
            source_type="csv",
            source_path=file_path,
            category=category,
            metadata={"file_size": os.path.getsize(file_path), "headers": headers},
        )

    def load_directory(self, directory_path: str, category: Optional[List[str]] = None):
        """Load all supported files from a directory"""
        directory = Path(directory_path)
        loaded_files = []

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    cat = (
                        category[0]
                        if isinstance(category, list) and category
                        else category if isinstance(category, str) else None
                    )
                    if file_path.suffix.lower() == ".md":
                        doc_id = self.load_markdown_file(str(file_path), cat)
                        loaded_files.append((str(file_path), doc_id))
                    elif file_path.suffix.lower() == ".json":
                        doc_id = self.load_json_file(str(file_path), category)
                        loaded_files.append((str(file_path), doc_id))
                    elif file_path.suffix.lower() == ".csv":
                        doc_id = self.load_csv_file(str(file_path), cat)
                        loaded_files.append((str(file_path), doc_id))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return loaded_files


def load_postgresql_knowledge_base():
    """Load knowledge base articles from PostgreSQL into the search index"""
    session = get_session()
    kb = SimpleKnowledgeBase()

    try:
        # Load knowledge base articles from PostgreSQL
        articles = session.query(KnowledgeBase).filter(KnowledgeBase.is_active).all()

        for article in articles:
            kb.add_document(
                title=getattr(article, "title", ""),
                content=getattr(article, "content", ""),
                source_type="postgresql",
                source_path=f"knowledge_base.id={getattr(article, 'id', '')}",
                category=getattr(article, "category", None),
                metadata={
                    "tags": getattr(article, "tags", None),
                    "created_at": (
                        article.created_at.isoformat()
                        if getattr(article, "created_at", None)
                        else None
                    ),
                    "updated_at": (
                        article.updated_at.isoformat()
                        if getattr(article, "updated_at", None)
                        else None
                    ),
                },
            )

        print(f"Loaded {len(articles)} articles from PostgreSQL")
        return len(articles)

    except Exception as e:
        print(f"Error loading from PostgreSQL: {e}")
        return 0
    finally:
        session.close()


def initialize_knowledge_base():
    """Initialize the knowledge base with all available data"""
    kb = SimpleKnowledgeBase()
    loader = DocumentLoader(kb)

    print("Initializing knowledge base...")

    # Load PostgreSQL knowledge base
    pg_count = load_postgresql_knowledge_base()

    # Load file-based documents
    data_dir = Path("data")

    # Load unstructured documents
    unstructured_files = loader.load_directory(
        str(data_dir / "unstructured"), ["documentation"]
    )
    print(f"Loaded {len(unstructured_files)} unstructured documents")

    # Load structured JSON files
    structured_files = loader.load_directory(
        str(data_dir / "structured"), ["configuration"]
    )
    print(f"Loaded {len(structured_files)} structured documents")

    # Load semi-structured CSV files
    semi_structured_files = loader.load_directory(
        str(data_dir / "semi_structured"), ["data"]
    )
    print(f"Loaded {len(semi_structured_files)} semi-structured documents")

    total_docs = (
        pg_count
        + len(unstructured_files)
        + len(structured_files)
        + len(semi_structured_files)
    )
    print(f"Knowledge base initialized with {total_docs} total documents")

    return kb


def search_knowledge_base(
    query: str, category: Optional[str] = None, limit: int = 5
) -> List[Dict]:
    """Search the knowledge base for relevant documents with robust error handling"""
    try:
        if not query or not query.strip():
            return []

        # Try to initialize knowledge base
        try:
            kb = SimpleKnowledgeBase()
        except Exception as e:
            print(f"Error initializing knowledge base: {e}")
            return _get_fallback_results(query, limit)

        if category:
            try:
                # Search within specific category
                all_docs = kb.get_documents_by_category(category)
                # Simple text matching for category-specific search
                results = []
                query_lower = query.lower()
                for doc in all_docs:
                    if (
                        query_lower in doc["content"].lower()
                        or query_lower in doc["title"].lower()
                    ):
                        results.append(doc)
                        if len(results) >= limit:
                            break
                return results
            except Exception as e:
                print(f"Error searching by category: {e}")
                return _get_fallback_results(query, limit)
        else:
            try:
                # Full-text search across all documents
                results = kb.search_documents(query, limit)
                if results:
                    return results
                else:
                    print(f"No results from vector search, using fallback for: {query}")
                    return _get_fallback_results(query, limit)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return _get_fallback_results(query, limit)

    except Exception as e:
        print(f"Critical error in search_knowledge_base: {e}")
        return _get_fallback_results(query, limit)


def _get_fallback_results(query: str, limit: int = 5) -> List[Dict]:
    """Generate fallback results when knowledge base search fails"""
    fallback_results = []

    # Generate topic-specific fallback content
    if "marketing" in query.lower():
        fallback_results = [
            {
                "id": "fallback_marketing_1",
                "title": "Marketing Campaign Best Practices",
                "content": f"Essential marketing strategies for {query}. Focus on audience segmentation, compelling content creation, and multi-channel distribution for maximum reach and engagement.",
                "source_type": "fallback",
                "source_path": "internal",
                "category": "marketing",
                "metadata": {"type": "fallback", "query": query},
                "created_at": "2025-08-02T00:00:00",
            },
            {
                "id": "fallback_marketing_2",
                "title": "Digital Marketing Metrics and Analytics",
                "content": f"Key performance indicators for {query} campaigns including conversion rates, customer acquisition costs, and return on investment metrics.",
                "source_type": "fallback",
                "source_path": "internal",
                "category": "analytics",
                "metadata": {"type": "fallback", "query": query},
                "created_at": "2025-08-02T00:00:00",
            },
        ]
    elif "customer" in query.lower() or "support" in query.lower():
        fallback_results = [
            {
                "id": "fallback_support_1",
                "title": "Customer Support Guidelines",
                "content": f"Customer support best practices for handling {query}. Emphasize empathy, active listening, and prompt resolution of customer issues.",
                "source_type": "fallback",
                "source_path": "internal",
                "category": "support",
                "metadata": {"type": "fallback", "query": query},
                "created_at": "2025-08-02T00:00:00",
            }
        ]
    else:
        # Generic fallback
        fallback_results = [
            {
                "id": "fallback_generic_1",
                "title": f"Information about {query}",
                "content": f"General information and best practices related to {query}. This content is generated as a fallback when the knowledge base is not available.",
                "source_type": "fallback",
                "source_path": "internal",
                "category": "general",
                "metadata": {"type": "fallback", "query": query},
                "created_at": "2025-08-02T00:00:00",
            }
        ]

    return fallback_results[:limit]


if __name__ == "__main__":
    # Initialize the knowledge base
    kb = initialize_knowledge_base()

    # Test search functionality
    print("\n--- Testing Knowledge Base Search ---")

    # Test searches
    test_queries = [
        "password reset",
        "billing issues",
        "API integration",
        "marketing campaign",
        "customer support",
    ]

    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = search_knowledge_base(query, limit=3)
        for i, result in enumerate(results, 1):
            print(
                f"  {i}. {result['title']} ({result['source_type']}) - {result['category']}"
            )
            print(f"     {result['content'][:100]}...")
