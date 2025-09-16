from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

class FinancialRetriever:
    """Retriever for financial documents with enhanced capabilities"""
    
    def __init__(self, vector_index, llm=None):
        self.vector_index = vector_index
        self.llm = llm or ChatOpenAI(
            temperature=0.1,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Financial-specific prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a financial credit intelligence assistant. Use the following context to answer the question about credit risk assessment.

Context: {context}

Question: {question}

Please provide a comprehensive answer focusing on:
1. Credit risk assessment
2. Financial patterns identified
3. Sector-specific considerations
4. Risk mitigation suggestions

Answer:""",
            input_variables=["context", "question"]
        )
    
    def retrieve_relevant_documents(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """Retrieve relevant financial documents"""
        return self.vector_index.search(query, k=k, filter_dict=filters)
    
    def generate_answer(self, query: str, context_documents: List[Document]) -> str:
        """Generate answer using LLM with retrieved context"""
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        
        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )
        
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def full_retrieval_qa(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Complete retrieval QA process"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(query, k=k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "query": query
        }
    
    def sector_specific_retrieval(self, query: str, sector: str, k: int = 5) -> Dict[str, Any]:
        """Retrieve documents specific to a financial sector"""
        filters = {"sector": sector} if sector else None
        retrieved_docs = self.retrieve_relevant_documents(query, k=k, filters=filters)
        
        answer = self.generate_answer(query, retrieved_docs)
        
        return {
            "answer": answer,
            "sector": sector,
            "source_documents": retrieved_docs,
            "query": query
        }

def create_financial_rag_system(index_path: str, llm=None) -> FinancialRetriever:
    """Create a financial RAG system from existing index"""
    index = FinancialVectorIndex(index_path=index_path)
    index.load_index()
    
    retriever = FinancialRetriever(vector_index=index, llm=llm)
    return retriever

# Utility functions for integration with your main app
def setup_financial_rag(knowledge_base_path: str, index_path: str) -> FinancialRetriever:
    """Setup complete financial RAG system"""
    from .loaders import load_knowledge_base
    from .chunking import chunk_financial_documents, optimize_chunks_for_retrieval
    from .index import build_financial_rag_index
    
    # Load documents
    documents = load_knowledge_base(knowledge_base_path)
    
    # Chunk documents
    chunks = chunk_financial_documents(documents, method="adaptive")
    optimized_chunks = optimize_chunks_for_retrieval(chunks)
    
    # Build index
    index = build_financial_rag_index(optimized_chunks, index_path)
    
    # Create retriever
    retriever = FinancialRetriever(vector_index=index)
    
    return retriever
