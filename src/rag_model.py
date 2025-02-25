import os
import re
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from typing import List, Dict, Tuple, Optional

# Import the GeminiLLM class if available
try:
    from llm_integration import GeminiLLM
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class GitLabRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_llm: bool = False, llm_model_name: str = "gemini-2.0-flash"):
        """
        Initialize the GitLab RAG model.
        
        Args:
            model_name: The name of the sentence transformer model to use
            use_llm: Whether to use the LLM for response generation
            llm_model_name: The name of the Gemini model to use (if use_llm is True)
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Initialize LLM if requested and available
        self.llm = None
        if use_llm and HAS_GEMINI:
            try:
                self.llm = GeminiLLM(model_name=llm_model_name)
                print(f"Initialized Gemini LLM for response generation with model: {llm_model_name}")
            except Exception as e:
                print(f"Failed to initialize Gemini LLM: {e}")
        
    def load_markdown_files(self, directory: str) -> None:
        """
        Load all markdown files from a directory and its subdirectories.
        
        Args:
            directory: The directory to search for markdown files
        """
        print(f"Loading markdown files from {directory}...")
        markdown_files = glob.glob(f"{directory}/**/*.md", recursive=True)
        
        for file_path in tqdm(markdown_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Extract title from frontmatter if available
                title_match = re.search(r'title: (.*)', content)
                title = title_match.group(1) if title_match else os.path.basename(file_path)
                
                # Process the content into chunks
                chunks = self._chunk_markdown(content)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadata.append({
                        'source': file_path,
                        'title': title,
                        'chunk_id': i
                    })
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        print(f"Loaded {len(self.documents)} chunks from {len(markdown_files)} files")
    
    def _chunk_markdown(self, content: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split markdown content into overlapping chunks.
        
        Args:
            content: The markdown content to split
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Remove frontmatter
        content = re.sub(r'---.*?---', '', content, flags=re.DOTALL)
        
        # Simple chunking by characters with overlap
        chunks = []
        for i in range(0, len(content), max_chunk_size - overlap):
            chunk = content[i:i + max_chunk_size]
            if len(chunk) > 100:  # Only keep chunks with substantial content
                chunks.append(chunk)
                
        return chunks
    
    def create_embeddings(self) -> None:
        """Generate embeddings for all loaded documents."""
        if not self.documents:
            print("No documents loaded. Please load documents first.")
            return
            
        print("Creating embeddings...")
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
        print(f"Created embeddings with shape {self.embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        if self.embeddings is None:
            print("No embeddings created. Please create embeddings first.")
            return []
            
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities - reshape query_embedding to 2D array for cosine_similarity
        query_embedding_2d = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding_2d, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': similarities[idx]
            })
            
        return results
    
    def get_reddit_posts(self, db_path: str = 'reddit_posts.db') -> List[Dict]:
        """
        Retrieve Reddit posts from the database.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            List of dictionaries containing post data
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, content, url FROM posts")
            posts = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': post[0],
                    'title': post[1],
                    'content': post[2],
                    'url': post[3]
                }
                for post in posts
            ]
        except Exception as e:
            print(f"Error retrieving Reddit posts: {e}")
            return []
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate a response based on retrieved documents.
        
        Args:
            query: The user query
            retrieved_docs: The retrieved documents
            
        Returns:
            Generated response
        """
        # Use LLM if available
        if self.llm is not None:
            try:
                return self.llm.rag_generate(query, retrieved_docs)
            except Exception as e:
                print(f"Error generating response with LLM: {e}")
                print("Falling back to template response")
        
        # Fallback to template response
        response = f"Query: {query}\n\n"
        response += "Based on the GitLab documentation, here's what I found:\n\n"
        
        for i, doc in enumerate(retrieved_docs):
            response += f"Reference {i+1} (from {doc['metadata']['title']}):\n"
            response += f"{doc['content'][:300]}...\n\n"
            
        return response
    
    def save_to_db(self, post_id: str, response: str, db_path: str = 'reddit_posts.db') -> None:
        """
        Save a generated response to the database.
        
        Args:
            post_id: The ID of the Reddit post
            response: The generated response
            db_path: Path to the SQLite database
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE posts SET draft_response = ? WHERE id = ?",
                (response, post_id)
            )
            conn.commit()
            conn.close()
            print(f"Saved response for post {post_id}")
        except Exception as e:
            print(f"Error saving response: {e}")


if __name__ == "__main__":
    # Example usage
    rag = GitLabRAG(use_llm=True)
    rag.load_markdown_files("knowledgebase")
    rag.create_embeddings()
    
    # Example query
    query = "How do I set up GitLab CI/CD pipelines with Docker?"
    results = rag.retrieve(query, top_k=3)
    
    print("\nQuery:", query)
    print("\nTop results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {result['metadata']['source']}")
        print(f"Title: {result['metadata']['title']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content preview: {result['content'][:200]}...")
    
    # Generate a response
    response = rag.generate_response(query, results)
    print("\nGenerated response:")
    print(response) 