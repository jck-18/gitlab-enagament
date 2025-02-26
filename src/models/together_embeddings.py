import os
import requests
import numpy as np
from typing import List, Dict, Optional, Union, Any
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

class TogetherEmbeddings:
    """
    A class for generating embeddings using Together AI's API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "togethercomputer/m2-bert-80M-8k-retrieval"):
        """
        Initialize the Together Embeddings.
        
        Args:
            api_key: Together API key. If None, will try to load from environment variable.
            model_name: The embedding model to use. Default is "togethercomputer/m2-bert-80M-8k-retrieval".
        """
        # Get API key from environment variable or config if not provided
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not self.api_key:
            # Try to load from config file
            try:
                config = self.load_config()
                self.api_key = config.get('together_api_key')
            except:
                pass
                
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass an API key to the constructor, "
                "set the TOGETHER_API_KEY environment variable, or add it to config/config.yaml."
            )
        
        # Set the model name
        self.model_name = model_name
        self.api_url = "https://api.together.xyz/v1/embeddings"
        
        print(f"Initialized Together Embeddings with model: {self.model_name}")
    
    def load_config(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding as a list of floats
        """
        embeddings = self.get_embeddings([text])
        if embeddings:
            return embeddings[0]
        return []
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(self, query_embedding: List[float], document_embeddings: List[List[float]], top_k: int = 3) -> List[int]:
        """
        Find the most similar documents to a query.
        
        Args:
            query_embedding: Embedding of the query
            document_embeddings: List of document embeddings
            top_k: Number of most similar documents to return
            
        Returns:
            List of indices of the most similar documents
        """
        similarities = [self.compute_similarity(query_embedding, doc_embedding) 
                        for doc_embedding in document_embeddings]
        
        # Get indices of top_k highest similarities
        return np.argsort(similarities)[-top_k:][::-1].tolist() 