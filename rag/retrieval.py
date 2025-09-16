"""
Document retrieval module for RAG system in CreditAI application.
Implements various retrieval strategies for credit-related queries.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from loaders import Document, DocumentLoader
from chunking import Chunk, DocumentChunker
from index import VectorIndex, CreditIndex, IndexedChunk

@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    chunk: Chunk
    score: float
    retrieval_method: str
    query: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class QueryProcessor:
    """Processes and analyzes user queries for optimal retrieval."""
    
    def __init__(self):
        self.credit_keywords = {
            'risk': ['risk', 'risky', 'dangerous', 'safe', 'secure'],
            'amount': ['amount', 'value', 'sum', 'total', 'money', 'dollar'],
            'balance': ['balance', 'remaining', 'account', 'funds'],
            'payment': ['payment', 'pay', 'transaction', 'transfer'],
            'score': ['score', 'rating', 'grade', 'assessment'],
            'sector': ['sector', 'industry', 'business', 'company', 'field'],
            'time': ['recent', 'new', 'old', 'latest', 'current', 'past']
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine retrieval strategy."""
        query_lower = query.lower()
        
        analysis = {
            'original_query': query,
            'query_type': 'general',
            'keywords': [],
            'numerical_values': [],
            'risk_keywords': [],
            'temporal_keywords': [],
            'suggested_method': 'semantic'
        }
        
        # Extract keywords by category
        for category, keywords in self.credit_keywords.items():
            found = [kw for kw in keywords if kw in query_lower]
            if found:
                analysis['keywords'].extend(found)
                if category == 'risk':
                    analysis['risk_keywords'].extend(found)
                elif category == 'time':
                    analysis['temporal_keywords'].extend(found)
        
        # Extract numerical values
        numbers = re.findall(r'\d+\.?\d*', query)
        analysis['numerical_values'] = [float(n) for n in numbers]
        
        # Determine query type and suggested method
        if analysis['risk_keywords']:
            analysis['query_type'] = 'risk_assessment'
            analysis['suggested_method'] = 'risk_based'
        elif analysis['numerical_values']:
            analysis['query_type'] = 'numerical_search'
            analysis['suggested_method'] = 'metadata_filter'
        elif any(kw in query_lower for kw in ['similar', 'like', 'compare']):
            analysis['query_type'] = 'similarity_search'
            analysis['suggested_method'] = 'similarity'
        elif any(kw in query_lower for kw in ['policy', 'rule', 'guideline']):
            analysis['query_type'] = 'policy_search'
            analysis['suggested_method'] = 'semantic'
        
        return analysis

class BaseRetriever:
    """Base class for retrieval strategies."""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        raise NotImplementedError

class SemanticRetriever(BaseRetriever):
    """Retrieves documents using semantic similarity."""
    
    def __init__(self, index: VectorIndex):
        self.index = index
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using TF-IDF semantic similarity."""
        results = []
        
        search_results = self.index.search(query, top_k)
        
        for indexed_chunk, score in search_results:
            results.append(RetrievalResult(
                chunk=indexed_chunk.chunk,
                score=score,
                retrieval_method="semantic_tfidf",
                query=query
            ))
        
        return results

class KeywordRetriever(BaseRetriever):
    """Retrieves documents using keyword matching."""
    
    def __init__(self, index: VectorIndex):
        self.index = index
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using keyword matching."""
        # Extract keywords from query
        query_words = query.lower().split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        search_results = self.index.search_by_keywords(keywords, top_k)
        
        results = []
        for indexed_chunk, score in search_results:
            results.append(RetrievalResult(
                chunk=indexed_chunk.chunk,
                score=score,
                retrieval_method="keyword_matching",
                query=query
            ))
        
        return results

class MetadataRetriever(BaseRetriever):
    """Retrieves documents using metadata filtering."""
    
    def __init__(self, index: VectorIndex):
        self.index = index
    
    def retrieve(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Retrieve using metadata filters."""
        if filters is None:
            # Try to extract filters from query
            filters = self._extract_filters_from_query(query)
        
        if not filters:
            return []
        
        search_results = self.index.search_by_metadata(filters, top_k)
        
        results = []
        for indexed_chunk in search_results:
            # Calculate a simple relevance score based on filter matches
            score = self._calculate_metadata_score(indexed_chunk.chunk.metadata, filters)
            
            results.append(RetrievalResult(
                chunk=indexed_chunk.chunk,
                score=score,
                retrieval_method="metadata_filter",
                query=query
            ))
        
        return results
    
    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from natural language query."""
        filters = {}
        query_lower = query.lower()
        
        # Look for risk level indicators
        if 'high risk' in query_lower:
            filters['Risk_Label'] = 1
        elif 'low risk' in query_lower:
            filters['Risk_Label'] = 0
        
        # Look for sector mentions
        sector_patterns = {
            'retail': 1, 'manufacturing': 2, 'services': 3, 'technology': 4,
            'healthcare': 5, 'finance': 6, 'education': 7, 'government': 8
        }
        
        for sector, code in sector_patterns.items():
            if sector in query_lower:
                filters['Sector_Code'] = code
                break
        
        # Look for numerical ranges
        import re
        amounts = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', query)
        if amounts:
            # This is a simple example - you might want more sophisticated range parsing
            amount = float(amounts[0].replace(',', ''))
            filters['Amount_range'] = amount
        
        return filters
    
    def _calculate_metadata_score(self, metadata: Dict, filters: Dict) -> float:
        """Calculate relevance score based on metadata matches."""
        matches = 0
        total_filters = len(filters)
        
        for key, value in filters.items():
            if key in metadata and metadata[key] == value:
                matches += 1
        
        return matches / total_filters if total_filters > 0 else 0

class CreditRetriever(BaseRetriever):
    """Specialized retriever for credit-related queries."""
    
    def __init__(self, credit_index: CreditIndex):
        self.credit_index = credit_index
        self.semantic_retriever = SemanticRetriever(credit_index.vector_index)
        self.keyword_retriever = KeywordRetriever(credit_index.vector_index)
        self.metadata_retriever = MetadataRetriever(credit_index.vector_index)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using credit-specific logic."""
        query_lower = query.lower()
        
        # Risk-based queries
        if 'risk' in query_lower:
            return self._retrieve_by_risk(query, top_k)
        
        # Similarity queries
        elif any(word in query_lower for word in ['similar', 'like', 'compare']):
            return self._retrieve_similar_profiles(query, top_k)
        
        # Amount/balance queries
        elif any(word in query_lower for word in ['amount', 'balance', 'money', 'dollar']):
            return self._retrieve_by_financial_criteria(query, top_k)
        
        # Default to semantic search
        else:
            return self.semantic_retriever.retrieve(query, top_k)
    
    def _retrieve_by_risk(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve documents by risk level."""
        results = []
        
        # Try to determine risk level from query
        if 'high' in query.lower():
            risk_chunks = self.credit_index.search_by_risk_level(1, top_k)
        elif 'low' in query.lower():
            risk_chunks = self.credit_index.search_by_risk_level(0, top_k)
        else:
            # Return both high and low risk examples
            high_risk = self.credit_index.search_by_risk_level(1, top_k // 2)
            low_risk = self.credit_index.search_by_risk_level(0, top_k - len(high_risk))
            risk_chunks = high_risk + low_risk
        
        for i, chunk in enumerate(risk_chunks):
            results.append(RetrievalResult(
                chunk=chunk,
                score=1.0 - (i * 0.1),  # Decreasing score by order
                retrieval_method="risk_based",
                query=query
            ))
        
        return results
    
    def _retrieve_similar_profiles(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve similar credit profiles."""
        # Extract numerical values from query for comparison
        import re
        numbers = re.findall(r'\d+\.?\d*', query)
        
        if numbers:
            # Create a metadata profile for similarity search
            query_metadata = {}
            
            # Simple heuristic: first number might be amount, second balance, etc.
            if len(numbers) >= 1:
                query_metadata['Amount'] = float(numbers[0])
            if len(numbers) >= 2:
                query_metadata['Balance'] = float(numbers[1])
            if len(numbers) >= 3:
                query_metadata['Punctuality_Score'] = float(numbers[2])
            
            search_results = self.credit_index.search_similar_credit_profiles(query_metadata, top_k)
            
            results = []
            for chunk, score in search_results:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    retrieval_method="similarity_search",
                    query=query
                ))
            
            return results
        else:
            # Fall back to semantic search
            return self.semantic_retriever.retrieve(query, top_k)
    
    def _retrieve_by_financial_criteria(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve by financial criteria like amount or balance."""
        # Extract numerical values
        import re
        amounts = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', query)
        
        if amounts:
            target_amount = float(amounts[0].replace(',', ''))
            
            # Find chunks with similar amounts or balances
            results = []
            
            for indexed_chunk in self.credit_index.vector_index.chunks:
                chunk = indexed_chunk.chunk
                metadata = chunk.metadata
                
                score = 0
                # Check amount similarity
                if 'Amount' in metadata:
                    amount_diff = abs(float(metadata['Amount']) - target_amount)
                    if amount_diff < target_amount * 0.2:  # Within 20%
                        score += 1.0 - (amount_diff / target_amount)
                
                # Check balance similarity
                if 'Balance' in metadata:
                    balance_diff = abs(float(metadata['Balance']) - target_amount)
                    if balance_diff < target_amount * 0.2:  # Within 20%
                        score += 0.5 * (1.0 - (balance_diff / target_amount))
                
                if score > 0:
                    results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        retrieval_method="financial_criteria",
                        query=query
                    ))
            
            # Sort by score and return top-k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        
        # If no amounts found, fall back to keyword search
        return self.keyword_retriever.retrieve(query, top_k)

class HybridRetriever(BaseRetriever):
    """Combines multiple retrieval strategies."""
    
    def __init__(self, credit_index: CreditIndex):
        self.credit_index = credit_index
        self.query_processor = QueryProcessor()
        self.credit_retriever = CreditRetriever(credit_index)
        self.semantic_retriever = SemanticRetriever(credit_index.vector_index)
        self.keyword_retriever = KeywordRetriever(credit_index.vector_index)
        self.metadata_retriever = MetadataRetriever(credit_index.vector_index)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using hybrid approach combining multiple strategies."""
        # Analyze query to determine best strategies
        analysis = self.query_processor.analyze_query(query)
        
        all_results = []
        
        # Apply primary strategy based on query analysis
        if analysis['suggested_method'] == 'risk_based':
            results = self.credit_retriever.retrieve(query, top_k)
            all_results.extend(results)
        
        elif analysis['suggested_method'] == 'similarity':
            results = self.credit_retriever._retrieve_similar_profiles(query, top_k)
            all_results.extend(results)
        
        elif analysis['suggested_method'] == 'metadata_filter':
            results = self.metadata_retriever.retrieve(query, top_k)
            all_results.extend(results)
        
        # Always include semantic results for diversity
        semantic_results = self.semantic_retriever.retrieve(query, top_k // 2)
        all_results.extend(semantic_results)
        
        # Remove duplicates and re-rank
        unique_results = self._remove_duplicates(all_results)
        
        # Re-rank results using combined score
        final_results = self._rerank_results(unique_results, analysis)
        
        return final_results[:top_k]
    
    def _remove_duplicates(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results based on chunk ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, results: List[RetrievalResult], analysis: Dict) -> List[RetrievalResult]:
        """Re-rank results using combined scoring."""
        for result in results:
            # Base score from retrieval method
            combined_score = result.score
            
            # Boost based on query analysis
            if analysis['query_type'] == 'risk_assessment' and result.retrieval_method == 'risk_based':
                combined_score *= 1.5
            
            elif analysis['query_type'] == 'numerical_search' and result.retrieval_method == 'metadata_filter':
                combined_score *= 1.3
            
            elif analysis['query_type'] == 'similarity_search' and result.retrieval_method == 'similarity_search':
                combined_score *= 1.4
            
            # Boost for keyword matches in content
            content_lower = result.chunk.content.lower()
            keyword_boost = sum(1 for kw in analysis['keywords'] if kw in content_lower) * 0.1
            combined_score += keyword_boost
            
            # Update the score
            result.score = combined_score
        
        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class RAGRetriever:
    """Main RAG retrieval interface for CreditAI application."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.credit_index = CreditIndex()
        self.hybrid_retriever = HybridRetriever(self.credit_index)
        self.is_initialized = False
    
    def initialize(self, document_paths: List[str] = None) -> bool:
        """Initialize the RAG system with documents."""
        try:
            # Load documents
            if document_paths is None:
                document_paths = self._find_credit_documents()
            
            if not document_paths:
                print("No documents found to initialize RAG system")
                return False
            
            loader = DocumentLoader()
            documents = loader.load_documents(document_paths)
            
            if not documents:
                print("No documents loaded")
                return False
            
            # Chunk documents
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(documents)
            
            if not chunks:
                print("No chunks created")
                return False
            
            # Index chunks
            self.credit_index.add_chunks(chunks)
            self.credit_index.build_index()
            
            self.is_initialized = True
            print(f"RAG system initialized with {len(chunks)} chunks from {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Query the RAG system."""
        if not self.is_initialized:
            print("RAG system not initialized. Call initialize() first.")
            return []
        
        try:
            results = self.hybrid_retriever.retrieve(question, top_k)
            print(f"Retrieved {len(results)} results for query: '{question}'")
            return results
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return []
    
    def _find_credit_documents(self) -> List[str]:
        """Find credit-related documents in the data directory."""
        import os
        import glob
        
        document_paths = []
        
        # Look for common file patterns
        patterns = [
            os.path.join(self.data_path, "*.csv"),
            os.path.join(self.data_path, "*.json"),
            os.path.join(self.data_path, "*.txt"),
            os.path.join(self.data_path, "credit*.csv"),
            os.path.join(self.data_path, "risk*.csv"),
            os.path.join(self.data_path, "policies", "*.txt")
        ]
        
        for pattern in patterns:
            document_paths.extend(glob.glob(pattern))
        
        return list(set(document_paths))  # Remove duplicates
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        stats = self.credit_index.vector_index.get_stats()
        stats["status"] = "initialized"
        stats["risk_levels"] = len(self.credit_index.risk_index)
        
        return stats
