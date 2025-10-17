"""
Pinecone Vector Database Client
Handles vector storage and semantic search for product recommendations
"""

import os
from typing import List, Dict, Optional
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

class PineconeClient:
    def __init__(self):
        """Initialize Pinecone client with API key from environment"""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "furniture-products")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", 384))
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
    def create_index(self):
        """Create a new Pinecone index if it doesn't exist"""
        try:
            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                print(f"✓ Created Pinecone index: {self.index_name}")
            else:
                print(f"✓ Index {self.index_name} already exists")
                
            self.index = self.pc.Index(self.index_name)
            return True
            
        except Exception as e:
            print(f"✗ Error creating index: {str(e)}")
            return False
    
    def upsert_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], batch_size: int = 100):
        """
        Upload product embeddings to Pinecone
        
        Args:
            embeddings: numpy array of shape (n_products, dimension)
            metadata: list of dictionaries containing product information
            batch_size: number of vectors to upload per batch
        """
        if self.index is None:
            self.create_index()
        
        vectors_to_upsert = []
        
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            vectors_to_upsert.append({
                "id": meta["uniq_id"],
                "values": embedding.tolist(),
                "metadata": meta
            })
            
            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size or i == len(embeddings) - 1:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        
        print(f"✓ Uploaded {len(embeddings)} embeddings to Pinecone")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar products using query embedding
        
        Args:
            query_embedding: query vector of shape (dimension,)
            top_k: number of results to return
            filter_dict: optional metadata filter
            
        Returns:
            List of matching products with scores
        """
        if self.index is None:
            self.create_index()
        
        # Perform search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })
        
        return matches
    
    def delete_index(self):
        """Delete the current index"""
        try:
            self.pc.delete_index(self.index_name)
            print(f"✓ Deleted index: {self.index_name}")
        except Exception as e:
            print(f"✗ Error deleting index: {str(e)}")
    
    def get_stats(self):
        """Get index statistics"""
        if self.index is None:
            self.create_index()
        
        stats = self.index.describe_index_stats()
        return stats


# Example usage
if __name__ == "__main__":
    client = PineconeClient()
    client.create_index()
    
    # Example: Create dummy embeddings
    n_products = 5
    dimension = 384
    dummy_embeddings = np.random.rand(n_products, dimension)
    
    dummy_metadata = [
        {
            "uniq_id": f"product_{i}",
            "title": f"Product {i}",
            "price": 100.0 + i * 10,
            "category": "furniture"
        }
        for i in range(n_products)
    ]
    
    # Upload embeddings
    client.upsert_embeddings(dummy_embeddings, dummy_metadata)
    
    # Search
    query_vector = np.random.rand(dimension)
    results = client.search(query_vector, top_k=3)
    print("Search results:", results)
    
    # Get stats
    stats = client.get_stats()
    print("Index stats:", stats)
