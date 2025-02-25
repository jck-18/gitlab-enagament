#!/usr/bin/env python3
"""
Test script for the GitLab RAG model.
This script demonstrates the RAG model with a sample query.
"""

import sys
from rag_model import GitLabRAG

def main():
    """Main function to test the RAG model."""
    # Sample query
    query = "How do I set up GitLab CI/CD pipelines with Docker?"
    
    # Check if a query was provided as a command-line argument
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    print(f"Testing RAG model with query: {query}")
    
    # Initialize the RAG model
    print("\nInitializing RAG model...")
    rag = GitLabRAG(use_llm=False)  # Set to True to use LLM if available
    
    # Load markdown files
    print("\nLoading markdown files...")
    rag.load_markdown_files("knowledgebase")
    
    # Create embeddings
    print("\nCreating embeddings...")
    rag.create_embeddings()
    
    # Retrieve relevant documents
    print("\nRetrieving relevant documents...")
    results = rag.retrieve(query, top_k=3)
    
    # Print results
    print("\nTop results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {result['metadata']['source']}")
        print(f"Title: {result['metadata']['title']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content preview: {result['content'][:200]}...")
    
    # Generate a response
    print("\nGenerating response...")
    response = rag.generate_response(query, results)
    
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
        prompt += f"Reference {i+1}: {doc['content'][:1000]}\n\n"
    
    prompt += f"User Query: \"{query}\"\n\n"
    prompt += "Instruction: Combine all references to create the best possible answer. If unsure, say so."
    
    return prompt

if __name__ == "__main__":
    main() 