"""
RAG Knowledge Base using ChromaDB for document retrieval
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import ast
import os
from typing import List, Dict, Any

class RAGKnowledgeBase:
    """RAG Knowledge Base for document retrieval"""
    
    def __init__(self, collection_name="document_extraction", persist_directory=None):
        """Initialize RAG knowledge base"""
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", "knowledge_base"
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"RAG Knowledge Base initialized: {self.collection_name}")
    
    def load_training_data(self):
        """Load processed training data"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_path = os.path.join(project_root, "data", "processed", "document_benchmark_processed.csv")
        
        df = pd.read_csv(data_path)
        df['labels'] = df['labels'].apply(ast.literal_eval)
        return df
    
    def prepare_documents_for_indexing(self, df):
        """Prepare documents for indexing in ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create document text from labels
            labels_text = json.dumps(row['labels'], indent=2)
            
            # Add metadata
            metadata = {
                'doc_type': row.get('doc_type', 'unknown'),
                'source_dataset': row.get('source_dataset', 'unknown'),
                'doc_id': row.get('doc_id', f'doc_{idx}')
            }
            
            documents.append(labels_text)
            metadatas.append(metadata)
            ids.append(str(idx))
        
        return documents, metadatas, ids
    
    def index_documents(self, df):
        """Index documents in ChromaDB"""
        print("Preparing documents for indexing...")
        documents, metadatas, ids = self.prepare_documents_for_indexing(df)
        
        print(f"Indexing {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        print(f"Successfully indexed {len(documents)} documents")
    
    def retrieve_similar_documents(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar documents based on query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs
    
    def get_collection_stats(self):
        """Get collection statistics"""
        count = self.collection.count()
        return {'document_count': count}

def main():
    """Build RAG knowledge base from training data"""
    print("Building RAG Knowledge Base...")
    
    # Initialize knowledge base
    kb = RAGKnowledgeBase()
    
    # Load training data
    print("Loading training data...")
    df = kb.load_training_data()
    print(f"Loaded {len(df)} documents")
    
    # Index documents
    kb.index_documents(df)
    
    # Show stats
    stats = kb.get_collection_stats()
    print(f"\nKnowledge Base Stats: {stats}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    test_query = json.dumps({'total_total_price': '100.00', 'doc_type': 'invoice'})
    results = kb.retrieve_similar_documents(test_query, n_results=3)
    
    print(f"Retrieved {len(results)} similar documents:")
    for i, doc in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Distance: {doc['distance']:.4f}")
        print(f"    Doc Type: {doc['metadata']['doc_type']}")
        print(f"    Source: {doc['metadata']['source_dataset']}")
        print(f"    Document: {doc['document'][:200]}...")

if __name__ == "__main__":
    main()
