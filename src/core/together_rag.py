import os
import re
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any

# Import chromadb conditionally to handle cases where it's not installed
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from together_llm import TogetherLLM
from together_embeddings import TogetherEmbeddings

class TogetherRAG:
    """
    A RAG (Retrieval-Augmented Generation) implementation using Together AI's API.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "togethercomputer/m2-bert-80M-8k-retrieval",
                 llm_model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 api_key: Optional[str] = None,
                 use_chroma: bool = True,
                 chroma_persist_directory: str = "./chroma_db"):
        """
        Initialize the Together RAG model.
        
        Args:
            embedding_model_name: The name of the Together embedding model to use
            llm_model_name: The name of the Together LLM model to use
            api_key: Together API key. If None, will try to load from environment variable.
            use_chroma: Whether to use ChromaDB for vector storage
            chroma_persist_directory: Directory to persist ChromaDB
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not self.api_key:
            # Try to load from config file
            try:
                with open('config/config.yaml', 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    self.api_key = config.get('together_api_key')
            except:
                pass
        
        # Initialize the embedding model
        self.embeddings = TogetherEmbeddings(api_key=self.api_key, model_name=embedding_model_name)
        
        # Initialize the LLM
        self.llm = TogetherLLM(api_key=self.api_key, model_name=llm_model_name)
        
        # Check if ChromaDB is available
        if use_chroma and not HAS_CHROMADB:
            print("Warning: ChromaDB is not installed. Falling back to in-memory storage.")
            use_chroma = False
        
        # Initialize ChromaDB if requested and available
        self.use_chroma = use_chroma
        if use_chroma:
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
            
            # Create a custom embedding function that uses Together API
            class TogetherEmbeddingFunction(chromadb.EmbeddingFunction):
                def __init__(self, embeddings_client, model_name):
                    self.embeddings_client = embeddings_client
                    self.model_name = model_name
                
                def __call__(self, texts):
                    return self.embeddings_client.get_embeddings(texts)
                
                def embed_documents(self, documents):
                    return self.embeddings_client.get_embeddings(documents)
                
                def embed_query(self, query):
                    return self.embeddings_client.get_embedding(query)
            
            # Use the custom class
            self.together_ef = TogetherEmbeddingFunction(self.embeddings, embedding_model_name)
            
            # Create or get the collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name="together_rag_collection",
                    embedding_function=self.together_ef
                )
                print(f"Using existing ChromaDB collection with {self.collection.count()} documents")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="together_rag_collection",
                    embedding_function=self.together_ef
                )
                print("Created new ChromaDB collection")
        else:
            # If not using ChromaDB, store documents and embeddings in memory
            self.documents = []
            self.document_embeddings = []
            self.metadata = []
    
    def load_markdown_files(self, directory: str) -> None:
        """
        Load all markdown files from a directory and its subdirectories.
        
        Args:
            directory: The directory to search for markdown files
        """
        print(f"Loading markdown files from {directory}...")
        markdown_files = glob.glob(f"{directory}/**/*.md", recursive=True)
        
        for file_path in tqdm(markdown_files, desc="Processing markdown files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Process the content into chunks
                chunks = self._chunk_markdown(content)
                
                # Get relative path for metadata
                rel_path = os.path.relpath(file_path, directory)
                
                # Add each chunk to the RAG system
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": rel_path,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                    self.add_document(chunk, metadata)
            
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    def _chunk_markdown(self, content: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split markdown content into overlapping chunks.
        
        Args:
            content: The markdown content to split
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of content chunks
        """
        # Simple chunking by character count with overlap
        chunks = []
        
        # Split by paragraphs first to avoid breaking in the middle of a paragraph
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max_chunk_size, save the current chunk and start a new one
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap from the end of the previous chunk
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document to the RAG system.
        
        Args:
            content: The document content
            metadata: Metadata about the document
        """
        if self.use_chroma:
            # Add to ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"doc_{self.collection.count()}"]
            )
        else:
            # Add to in-memory storage
            self.documents.append(content)
            embedding = self.embeddings.get_embedding(content)
            self.document_embeddings.append(embedding)
            self.metadata.append(metadata)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The query to search for
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        if self.use_chroma:
            # Retrieve from ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            retrieved_docs = []
            # Check if results and documents exist
            if results and "documents" in results and results["documents"]:
                for i in range(len(results["documents"])):
                    doc_data = {
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i] if "metadatas" in results and results["metadatas"] else {},
                        "id": results["ids"][i] if "ids" in results and results["ids"] else f"doc_{i}"
                    }
                    if "distances" in results and results["distances"]:
                        doc_data["distance"] = results["distances"][i]
                    retrieved_docs.append(doc_data)
            
            return retrieved_docs
        else:
            # Retrieve from in-memory storage
            if not self.document_embeddings:
                print("No documents have been added yet.")
                return []
            
            query_embedding = self.embeddings.get_embedding(query)
            top_indices = self.embeddings.find_most_similar(query_embedding, self.document_embeddings, top_k)
            
            retrieved_docs = []
            for idx in top_indices:
                retrieved_docs.append({
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "id": f"doc_{idx}",
                    "similarity": self.embeddings.compute_similarity(query_embedding, self.document_embeddings[idx])
                })
            
            return retrieved_docs
    
    def generate_response(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate a response for a query using RAG.
        
        Args:
            query: The query to answer
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the response and retrieved documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Generate response using the LLM
        response = self.llm.rag_generate(query, retrieved_docs)
        
        return {
            "query": query,
            "response": response,
            "retrieved_documents": retrieved_docs
        }
    
    def save_to_json(self, filename: str = "together_rag_results.json") -> None:
        """
        Save the RAG system state to a JSON file.
        
        Args:
            filename: The filename to save to
        """
        if self.use_chroma:
            print("ChromaDB is already persisted to disk. No need to save separately.")
            return
        
        data = {
            "documents": self.documents,
            "metadata": self.metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved RAG system state to {filename}")
    
    def load_from_json(self, filename: str = "together_rag_results.json") -> None:
        """
        Load the RAG system state from a JSON file.
        
        Args:
            filename: The filename to load from
        """
        if self.use_chroma:
            print("ChromaDB is already loaded from disk. No need to load separately.")
            return
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        
        # Regenerate embeddings
        self.document_embeddings = []
        for doc in tqdm(self.documents, desc="Generating embeddings"):
            embedding = self.embeddings.get_embedding(doc)
            self.document_embeddings.append(embedding)
        
        print(f"Loaded RAG system state from {filename} with {len(self.documents)} documents") 