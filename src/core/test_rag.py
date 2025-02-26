#!/usr/bin/env python3
"""
Test script for the GitLab RAG model.
This script demonstrates the RAG model with a sample query.

Usage:
    python test_rag.py [query] [--knowledge_dir DIR] [--top_k N] [--embedding_model MODEL] [--llm_model MODEL] [--skip_loading] [--chroma_dir DIR]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Union
from ..models.together_llm import TogetherLLM
from .together_rag import TogetherRAG
import traceback

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the GitLab RAG model.")
    parser.add_argument("query", nargs="?", default="How do I set up GitLab CI/CD pipelines with Docker?",
                        help="The query to test (default: 'How do I set up GitLab CI/CD pipelines with Docker?')")
    parser.add_argument("--knowledge_dir", "-k", default="../knowledgebase",
                        help="Directory containing knowledge base documents (default: '../knowledgebase')")
    parser.add_argument("--top_k", "-n", type=int, default=5,
                        help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--embedding_model", "-e", default="togethercomputer/m2-bert-80M-8k-retrieval",
                        help="Embedding model to use (default: 'togethercomputer/m2-bert-80M-8k-retrieval')")
    parser.add_argument("--llm_model", "-l", default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                        help="LLM model to use (default: 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free')")
    parser.add_argument("--skip_loading", "-s", action="store_true",
                        help="Skip loading documents if vectors are already stored")
    parser.add_argument("--chroma_dir", "-c", default="./chroma_db",
                        help="Directory to store ChromaDB persistence (default: './chroma_db')")
    return parser.parse_args()

def get_knowledge_dir(knowledge_dir):
    """Get the correct knowledge directory path."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Try different possible locations
    possible_paths = [
        knowledge_dir,
        os.path.join(workspace_root, 'knowledgebase'),
        os.path.join(workspace_root, knowledge_dir.lstrip('.').lstrip('/').lstrip('\\'))
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise ValueError(f"Knowledge directory not found in any of the expected locations: {possible_paths}")

def get_chroma_dir(chroma_dir: Optional[str]) -> str:
    """Get the correct chroma directory path."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Try different possible locations
    possible_paths = [
        chroma_dir if chroma_dir else os.path.join(workspace_root, 'chroma_db'),
        os.path.join(workspace_root, 'chroma_db'),
        os.path.join(workspace_root, chroma_dir.lstrip('.').lstrip('/').lstrip('\\')) if chroma_dir else None
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    # If no existing path found, return the default
    return os.path.join(workspace_root, 'chroma_db')

def main():
    """Main function to test the RAG model."""
    try:
        # Parse command-line arguments
        args = parse_args()
        query = args.query
        knowledge_dir = get_knowledge_dir(args.knowledge_dir)
        top_k = args.top_k
        embedding_model = args.embedding_model
        llm_model = args.llm_model
        skip_loading = args.skip_loading
        chroma_dir = get_chroma_dir(args.chroma_dir)
        
        # Check if we're running from the src directory and need to adjust paths
        if os.path.basename(os.getcwd()) == 'src' and knowledge_dir == '../knowledgebase':
            # We're in the src directory, so the default path is correct
            pass
        elif not os.path.exists(knowledge_dir) and os.path.exists('../knowledgebase'):
            # Try the parent directory
            knowledge_dir = '../knowledgebase'
            print(f"Knowledge directory not found, using '../knowledgebase' instead.")
        elif not os.path.exists(knowledge_dir) and os.path.exists('knowledgebase'):
            # Try the current directory
            knowledge_dir = 'knowledgebase'
            print(f"Knowledge directory not found, using 'knowledgebase' instead.")
        
        # Check for existing ChromaDB in the src directory
        if os.path.exists('./chroma_db') and chroma_dir == "./chroma_db":
            # Use the existing ChromaDB in the src directory
            print("Using existing ChromaDB in the src directory.")
            # Set skip_loading to True by default if using existing ChromaDB
            if not skip_loading:
                print("Setting --skip_loading to True since we're using an existing ChromaDB.")
                skip_loading = True
        
        # Create chroma_dir if it doesn't exist
        os.makedirs(chroma_dir, exist_ok=True)
        
        print(f"Testing RAG model with query: {query}")
        print(f"Knowledge directory: {knowledge_dir}")
        print(f"ChromaDB directory: {chroma_dir}")
        print(f"Top K: {top_k}")
        print(f"Embedding model: {embedding_model}")
        print(f"LLM model: {llm_model}")
        print(f"Skip loading documents: {skip_loading}")
        
        # Initialize the RAG model
        print("\nInitializing RAG model...")
        
        # Check for API key
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("Warning: TOGETHER_API_KEY environment variable not found.")
            print("Make sure you have set up your API key in the .env file or as an environment variable.")
        
        # Initialize the TogetherRAG model
        try:
            rag_model = TogetherRAG(
                embedding_model_name=embedding_model,
                llm_model_name=llm_model,
                use_chroma=True,
                chroma_persist_directory=chroma_dir
            )
        except Exception as e:
            print(f"Error initializing RAG model: {e}")
            return
        
        # Check if we should load documents
        if not skip_loading:
            # Load documents from the knowledgebase directory
            print(f"\nLoading documents from {knowledge_dir}...")
            
            # Check if the knowledgebase directory exists
            if not Path(knowledge_dir).exists():
                print(f"Warning: Knowledge directory '{knowledge_dir}' not found.")
                print("Please make sure the directory exists and contains markdown files.")
                return
            
            try:
                rag_model.load_markdown_files(knowledge_dir)
                print("Documents loaded and vectors stored in ChromaDB.")
            except Exception as e:
                print(f"Error loading documents: {e}")
                return
        else:
            print("\nSkipping document loading, using existing vectors from ChromaDB.")
            # Check if there are documents in the collection
            if hasattr(rag_model, 'collection') and rag_model.collection:
                try:
                    count = rag_model.collection.count()
                    print(f"Using existing ChromaDB collection with {count} documents.")
                    if count == 0:
                        print("Warning: ChromaDB collection is empty. Consider running without --skip_loading.")
                except Exception as e:
                    print(f"Error checking ChromaDB collection: {e}")
        
        # Retrieve relevant documents
        print("\nRetrieving relevant documents...")
        try:
            results = rag_model.retrieve(query, top_k=top_k)
            
            if not results:
                print("No relevant documents found. Please check your knowledgebase or try a different query.")
                return
                
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return
        
        # Print results
        print("\nTop results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            # Safely access metadata fields
            if isinstance(result, dict) and 'metadata' in result:
                metadata = result['metadata']
                if isinstance(metadata, dict):
                    print(f"Source: {metadata.get('source', 'N/A')}")
                    print(f"Title: {metadata.get('title', 'N/A')}")
                else:
                    print(f"Metadata: {metadata}")
            elif isinstance(result, dict):
                # If metadata is not available, print what we have
                for key, value in result.items():
                    if key != 'content':
                        print(f"{key}: {value}")
            
            # Print similarity if available
            if isinstance(result, dict) and 'similarity' in result:
                print(f"Similarity: {result['similarity']:.4f}")
            
            # Print content preview
            if isinstance(result, dict) and 'content' in result:
                content = result['content']
                if isinstance(content, str):
                    print(f"Content preview: {content[:200]}...")
                else:
                    print(f"Content type: {type(content)}")
            else:
                print(f"Result structure: {type(result)}")
        
        # Generate a response
        print("\nGenerating response...")
        try:
            response_data = rag_model.generate_response(query, top_k=top_k)
            
            # Handle different response formats
            if isinstance(response_data, str):
                response = response_data
            elif isinstance(response_data, dict) and 'response' in response_data:
                response = response_data['response']
            elif isinstance(response_data, dict):
                # If 'response' key is not available, use the first string value we find
                for key, value in response_data.items():
                    if isinstance(value, str) and len(value) > 0:
                        response = value
                        break
                else:
                    response = f"Response data structure: {response_data}"
            else:
                response = f"Unexpected response type: {type(response_data)}"
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return
        
        print("\nGenerated response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        # Format a prompt for an LLM
        prompt = format_prompt(query, results)
        
        print("\nFormatted prompt for LLM:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

def format_prompt(query, retrieved_docs):
    """
    Format a prompt for an LLM based on the retrieved documents.
    
    Args:
        query: The user query
        retrieved_docs: The retrieved documents
        
    Returns:
        Formatted prompt
    """
    prompt = "You are a helpful GitLab documentation assistant. "
    prompt += "Use the below references to answer the question accurately:\n\n"
    
    for i, doc in enumerate(retrieved_docs):
        if isinstance(doc, dict) and 'content' in doc:
            content = doc['content']
            if isinstance(content, str):
                prompt += f"Reference {i+1}: {content[:1000]}\n\n"
            else:
                prompt += f"Reference {i+1}: [Content not in text format]\n\n"
        else:
            prompt += f"Reference {i+1}: [Document format not recognized]\n\n"
    
    prompt += f"User Query: \"{query}\"\n\n"
    prompt += "Instruction: Combine all references to create the best possible answer. If unsure, say so."
    
    return prompt

if __name__ == "__main__":
    main() 