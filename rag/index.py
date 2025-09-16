import os
import pickle
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class FinancialVectorIndex:
    """Vector index for financial document retrieval"""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.documents = []
        
    def create_index(self, documents: List[Document]):
        """Create vector index from financial documents"""
        print(f"Creating financial vector index with {len(documents)} documents...")
        
        self.documents = documents
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        print(f"Index created with {self.vector_store.index.ntotal} vectors")
    
    def save_index(self):
        """Save index to disk"""
        if self.vector_store is None:
            raise ValueError("Index not created yet")
            
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.vector_store.save_local(str(self.index_path))
        
        # Save documents metadata
        documents_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in self.documents
        ]
        
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump(documents_data, f)
        
        # Save configuration
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'created_at': datetime.now().isoformat(),
            'num_vectors': self.vector_store.index.ntotal,
            'index_type': 'FAISS'
        }
        
        with open(self.index_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self):
        """Load index from disk"""
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError("Index file not found")
        
        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self.embedding_model
        )
        
        # Load documents metadata
        documents_file = self.index_path / "documents.pkl"
        if documents_file.exists():
            with open(documents_file, "rb") as f:
                documents_data = pickle.load(f)
            
            self.documents = [
                Document(page_content=data["page_content"], metadata=data["metadata"])
                for data in documents_data
            ]
        
        print(f"Index loaded with {self.vector_store.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar financial documents"""
        if self.vector_store is None:
            self.load_index()
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Prepare detailed results
        detailed_results = []
        for i, doc in enumerate(results):
            detailed_results.append({
                'rank': i + 1,
                'score': self._calculate_similarity_score(query, doc.page_content),
                'text': doc.page_content,
                'metadata': doc.metadata,
                'source': doc.metadata.get('source', 'unknown')
            })
        
        return detailed_results
    
    def _calculate_similarity_score(self, query: str, text: str) -> float:
        """Calculate similarity score between query and text"""
        query_embedding = self.embedding_model.embed_query(query)
        text_embedding = self.embedding_model.embed_query(text)
        
        similarity = np.dot(query_embedding, text_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
        )
        
        return float(similarity)

class MultiIndexManager:
    """Manager for multiple financial indices"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.indices = {}
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_index(self, index_name: str, documents: List[Document]):
        """Create a new index"""
        index_path = self.base_path / index_name
        index = FinancialVectorIndex(str(index_path))
        index.create_index(documents)
        index.save_index()
        self.indices[index_name] = index
        return index
    
    def load_index(self, index_name: str):
        """Load an existing index"""
        index_path = self.base_path / index_name
        index = FinancialVectorIndex(str(index_path))
        index.load_index()
        self.indices[index_name] = index
        return index
    
    def get_index(self, index_name: str) -> Optional[FinancialVectorIndex]:
        """Get an index by name"""
        if index_name in self.indices:
            return self.indices[index_name]
        
        # Try to load if not in memory
        index_path = self.base_path / index_name
        if index_path.exists():
            return self.load_index(index_name)
        
        return None
    
    def list_indices(self) -> List[str]:
        """List all available indices"""
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

def build_financial_rag_index(documents: List[Document], index_path: str) -> FinancialVectorIndex:
    """Build and save a financial RAG index"""
    index = FinancialVectorIndex(index_path=index_path)
    index.create_index(documents)
    index.save_index()
    return index
