"""
Document indexing module for RAG system in CreditAI application.
Implements vector indexing and storage for efficient retrieval.
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

from chunking import Chunk

@dataclass
class IndexedChunk:
    """Chunk with additional indexing information."""
    chunk: Chunk
    embedding: Optional[np.ndarray] = None
    tfidf_vector: Optional[np.ndarray] = None
    keywords: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.keywords is None:
            self.keywords = []

class VectorIndex:
    """Vector-based index for document chunks."""
    
    def __init__(self, index_path: str = "data/vector_index"):
        self.index_path = index_path
        self.chunks: List[IndexedChunk] = []
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.tfidf_matrix = None
        self.is_fitted = False
        
        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        new_indexed_chunks = []
        
        for chunk in chunks:
            # Extract keywords for credit-specific indexing
            keywords = self._extract_keywords(chunk.content)
            
            indexed_chunk = IndexedChunk(
                chunk=chunk,
                keywords=keywords
            )
            new_indexed_chunks.append(indexed_chunk)
        
        self.chunks.extend(new_indexed_chunks)
        print(f"Added {len(new_indexed_chunks)} chunks to index")
    
    def build_index(self) -> None:
        """Build TF-IDF vectors for all chunks."""
        if not self.chunks:
            print("No chunks to index")
            return
        
        # Extract text content from chunks
        texts = [chunk.chunk.content for chunk in self.chunks]
        
        # Fit TF-IDF vectorizer and transform texts
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Store TF-IDF vectors in indexed chunks
        for i, indexed_chunk in enumerate(self.chunks):
            indexed_chunk.tfidf_vector = self.tfidf_matrix[i].toarray().flatten()
        
        self.is_fitted = True
        print(f"Built TF-IDF index for {len(self.chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        """Search for relevant chunks using TF-IDF similarity."""
        if not self.is_fitted:
            print("Index not built. Building now...")
            self.build_index()
        
        if not self.chunks:
            return []
        
        # Transform query using fitted vectorizer
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return relevant results
                results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        """Search chunks by keyword matching."""
        results = []
        
        for chunk in self.chunks:
            score = 0
            chunk_keywords = set(kw.lower() for kw in chunk.keywords)
            query_keywords = set(kw.lower() for kw in keywords)
            
            # Calculate keyword overlap score
            if chunk_keywords and query_keywords:
                intersection = chunk_keywords.intersection(query_keywords)
                union = chunk_keywords.union(query_keywords)
                score = len(intersection) / len(union) if union else 0
            
            # Also check content for keyword presence
            content_lower = chunk.chunk.content.lower()
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 0.1  # Boost score for content matches
            
            if score > 0:
                results.append((chunk, score))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_by_metadata(self, metadata_filters: Dict[str, Any], top_k: int = 10) -> List[IndexedChunk]:
        """Search chunks by metadata criteria."""
        results = []
        
        for chunk in self.chunks:
            match = True
            for key, value in metadata_filters.items():
                chunk_value = chunk.chunk.metadata.get(key)
                if chunk_value != value:
                    match = False
                    break
            
            if match:
                results.append(chunk)
                if len(results) >= top_k:
                    break
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords relevant to credit analysis."""
        credit_keywords = [
            'risk', 'credit', 'score', 'balance', 'amount', 'payment', 'default',
            'interest', 'loan', 'debt', 'income', 'asset', 'liability', 'collateral',
            'guarantee', 'security', 'financial', 'bank', 'transaction', 'account',
            'punctuality', 'delinquent', 'overdue', 'sector', 'industry', 'business'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in credit_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Also extract numeric values as potential keywords
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        for num in numbers[:5]:  # Limit to first 5 numbers
            if float(num) > 0:
                found_keywords.append(f"value_{num}")
        
        return found_keywords
    
    def save_index(self, filename: str = None) -> str:
        """Save the index to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"credit_index_{timestamp}.pkl"
        
        filepath = os.path.join(self.index_path, filename)
        
        # Prepare data for saving
        index_data = {
            'chunks': [self._serialize_indexed_chunk(chunk) for chunk in self.chunks],
            'vectorizer': self.tfidf_vectorizer if self.is_fitted else None,
            'is_fitted': self.is_fitted,
            'created_at': datetime.now().isoformat(),
            'chunk_count': len(self.chunks)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index saved to {filepath}")
        return filepath
    
    def load_index(self, filename: str) -> bool:
        """Load index from disk."""
        filepath = os.path.join(self.index_path, filename) if not os.path.isabs(filename) else filename
        
        if not os.path.exists(filepath):
            print(f"Index file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            # Restore chunks
            self.chunks = [self._deserialize_indexed_chunk(chunk_data) for chunk_data in index_data['chunks']]
            
            # Restore vectorizer
            if index_data.get('vectorizer'):
                self.tfidf_vectorizer = index_data['vectorizer']
                # Rebuild TF-IDF matrix
                if self.chunks:
                    texts = [chunk.chunk.content for chunk in self.chunks]
                    self.tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            self.is_fitted = index_data.get('is_fitted', False)
            
            print(f"Loaded index with {len(self.chunks)} chunks from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def _serialize_indexed_chunk(self, indexed_chunk: IndexedChunk) -> Dict:
        """Serialize an IndexedChunk for storage."""
        return {
            'chunk': {
                'content': indexed_chunk.chunk.content,
                'metadata': indexed_chunk.chunk.metadata,
                'chunk_id': indexed_chunk.chunk.chunk_id,
                'start_index': indexed_chunk.chunk.start_index,
                'end_index': indexed_chunk.chunk.end_index
            },
            'keywords': indexed_chunk.keywords,
            'created_at': indexed_chunk.created_at
        }
    
    def _deserialize_indexed_chunk(self, data: Dict) -> IndexedChunk:
        """Deserialize an IndexedChunk from storage."""
        chunk = Chunk(
            content=data['chunk']['content'],
            metadata=data['chunk']['metadata'],
            chunk_id=data['chunk']['chunk_id'],
            start_index=data['chunk']['start_index'],
            end_index=data['chunk']['end_index']
        )
        
        return IndexedChunk(
            chunk=chunk,
            keywords=data.get('keywords', []),
            created_at=data.get('created_at')
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.chunks:
            return {"message": "Index is empty"}
        
        total_chunks = len(self.chunks)
        avg_content_length = np.mean([len(chunk.chunk.content) for chunk in self.chunks])
        
        # Count chunks by type
        chunk_types = {}
        for chunk in self.chunks:
            chunk_type = chunk.chunk.metadata.get('data_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Most common keywords
        all_keywords = []
        for chunk in self.chunks:
            all_keywords.extend(chunk.keywords)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_chunks": total_chunks,
            "average_content_length": round(avg_content_length, 2),
            "chunk_types": chunk_types,
            "top_keywords": top_keywords,
            "is_fitted": self.is_fitted,
            "vectorizer_features": self.tfidf_vectorizer.get_feature_names_out().shape[0] if self.is_fitted else 0
        }

class CreditIndex:
    """Specialized index for credit-related documents."""
    
    def __init__(self, index_path: str = "data/credit_index"):
        self.vector_index = VectorIndex(index_path)
        self.metadata_index = {}  # Quick lookup by metadata
        self.risk_index = {}  # Index by risk levels
        
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with credit-specific indexing."""
        self.vector_index.add_chunks(chunks)
        
        # Build metadata and risk indices
        for chunk in chunks:
            # Index by risk level
            risk_label = chunk.metadata.get('Risk_Label')
            if risk_label is not None:
                if risk_label not in self.risk_index:
                    self.risk_index[risk_label] = []
                self.risk_index[risk_label].append(chunk.chunk_id)
            
            # Index by sector
            sector = chunk.metadata.get('Sector_Code')
            if sector is not None:
                sector_key = f"sector_{sector}"
                if sector_key not in self.metadata_index:
                    self.metadata_index[sector_key] = []
                self.metadata_index[sector_key].append(chunk.chunk_id)
    
    def search_by_risk_level(self, risk_level: int, top_k: int = 10) -> List[Chunk]:
        """Search for chunks by risk level."""
        chunk_ids = self.risk_index.get(risk_level, [])
        
        # Find actual chunks
        results = []
        for indexed_chunk in self.vector_index.chunks:
            if indexed_chunk.chunk.chunk_id in chunk_ids and len(results) < top_k:
                results.append(indexed_chunk.chunk)
        
        return results
    
    def search_similar_credit_profiles(self, query_metadata: Dict[str, Any], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Find similar credit profiles based on metadata."""
        results = []
        
        for indexed_chunk in self.vector_index.chunks:
            chunk = indexed_chunk.chunk
            similarity = self._calculate_metadata_similarity(query_metadata, chunk.metadata)
            
            if similarity > 0:
                results.append((chunk, similarity))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _calculate_metadata_similarity(self, query_meta: Dict, chunk_meta: Dict) -> float:
        """Calculate similarity between metadata dictionaries."""
        score = 0
        total_fields = 0
        
        # Compare numerical fields
        numerical_fields = ['Amount', 'Balance', 'Punctuality_Score', 'Sector_Code']
        for field in numerical_fields:
            if field in query_meta and field in chunk_meta:
                total_fields += 1
                query_val = float(query_meta[field])
                chunk_val = float(chunk_meta[field])
                
                # Normalize similarity based on field type
                if field in ['Amount', 'Balance']:
                    # For monetary values, use relative difference
                    max_val = max(abs(query_val), abs(chunk_val))
                    if max_val > 0:
                        diff = abs(query_val - chunk_val) / max_val
                        score += max(0, 1 - diff)
                elif field == 'Punctuality_Score':
                    # For scores, direct similarity
                    diff = abs(query_val - chunk_val)
                    score += max(0, 1 - diff)
                elif field == 'Sector_Code':
                    # Exact match for sector
                    score += 1 if query_val == chunk_val else 0
        
        return score / total_fields if total_fields > 0 else 0
    
    def build_index(self) -> None:
        """Build all indices."""
        self.vector_index.build_index()
        print(f"Built credit index with {len(self.risk_index)} risk levels")
