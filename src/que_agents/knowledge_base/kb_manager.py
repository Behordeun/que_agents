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
from typing import Dict, List

import yaml
from src.que_agents.core.database import KnowledgeBase, get_session

# Load knowledge base configuration
with open("configs/knowledge_base_config.yaml", "r") as f:
    kb_config = yaml.safe_load(f)


class SimpleKnowledgeBase:
    """Simple knowledge base implementation using SQLite for vector storage simulation"""

    def __init__(self):
        self.db_path = kb_config["knowledge_base"]["db_path"]
        self.init_db()

    def init_db(self):
        """Initialize the knowledge base database"""
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

        # Create simple search index
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, content, category, content=documents, content_rowid=id
            )
        """
        )

        conn.commit()
        conn.close()

    def add_document(
        self,
        title: str,
        content: str,
        source_type: str,
        source_path: str = None,
        category: str = None,
        metadata: Dict = None,
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

        # Add to FTS index
        cursor.execute(
            """
            INSERT INTO documents_fts (rowid, title, content, category)
            VALUES (?, ?, ?, ?)
        """,
            (doc_id, title, content, category or ""),
        )

        conn.commit()
        conn.close()

        return doc_id

    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents using full-text search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sanitize query for FTS
        sanitized_query = (
            query.replace("'", "")
            .replace('"', "")
            .replace("!", "")
            .replace("?", "")
            .replace(".", "")
        )
        sanitized_query = " ".join(sanitized_query.split())  # Remove extra spaces

        if not sanitized_query.strip():
            sanitized_query = "help"

        try:
            # Use FTS for search
            cursor.execute(
                """
                SELECT d.id, d.title, d.content, d.source_type, d.source_path, 
                       d.category, d.metadata, d.created_at
                FROM documents_fts fts
                JOIN documents d ON fts.rowid = d.id
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (sanitized_query, limit),
            )
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS fails
            cursor.execute(
                """
                SELECT id, title, content, source_type, source_path, 
                       category, metadata, created_at
                FROM documents
                WHERE title LIKE ? OR content LIKE ?
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", limit),
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


class DocumentLoader:
    """Load documents from various sources into the knowledge base"""

    def __init__(self, knowledge_base: SimpleKnowledgeBase):
        self.kb = knowledge_base

    def load_markdown_file(self, file_path: str, category: str = None):
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

    def load_json_file(self, file_path: str, category: str = None):
        """Load a JSON file into the knowledge base"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert JSON to readable text
        content = json.dumps(data, indent=2)
        title = Path(file_path).stem

        return self.kb.add_document(
            title=title,
            content=content,
            source_type="json",
            source_path=file_path,
            category=category,
            metadata={
                "file_size": os.path.getsize(file_path),
                "json_keys": list(data.keys()) if isinstance(data, dict) else [],
            },
        )

    def load_csv_file(self, file_path: str, category: str = None):
        """Load a CSV file into the knowledge base"""
        content_lines = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            content_lines.append(f"CSV Headers: {', '.join(headers)}")
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

    def load_directory(self, directory_path: str, category: str = None):
        """Load all supported files from a directory"""
        directory = Path(directory_path)
        loaded_files = []

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == ".md":
                        doc_id = self.load_markdown_file(str(file_path), category)
                        loaded_files.append((str(file_path), doc_id))
                    elif file_path.suffix.lower() == ".json":
                        doc_id = self.load_json_file(str(file_path), category)
                        loaded_files.append((str(file_path), doc_id))
                    elif file_path.suffix.lower() == ".csv":
                        doc_id = self.load_csv_file(str(file_path), category)
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
        articles = (
            session.query(KnowledgeBase).filter(KnowledgeBase.is_active == True).all()
        )

        for article in articles:
            kb.add_document(
                title=article.title,
                content=article.content,
                source_type="postgresql",
                source_path=f"knowledge_base.id={article.id}",
                category=article.category,
                metadata={
                    "tags": article.tags,
                    "created_at": (
                        article.created_at.isoformat() if article.created_at else None
                    ),
                    "updated_at": (
                        article.updated_at.isoformat() if article.updated_at else None
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
        str(data_dir / "unstructured"), "documentation"
    )
    print(f"Loaded {len(unstructured_files)} unstructured documents")

    # Load structured JSON files
    structured_files = loader.load_directory(
        str(data_dir / "structured"), "configuration"
    )
    print(f"Loaded {len(structured_files)} structured documents")

    # Load semi-structured CSV files
    semi_structured_files = loader.load_directory(
        str(data_dir / "semi_structured"), "data"
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
    query: str, category: str = None, limit: int = 5
) -> List[Dict]:
    """Search the knowledge base for relevant documents"""
    kb = SimpleKnowledgeBase()

    if category:
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
    else:
        # Full-text search across all documents
        return kb.search_documents(query, limit)


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
