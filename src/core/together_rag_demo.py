#!/usr/bin/env python3
"""
Together RAG Demo Script

This script demonstrates how to use the Together AI API for implementing a RAG pipeline.
It follows the steps from the Together AI documentation:
https://docs.together.ai/docs/quickstart-retrieval-augmented-generation-rag
"""

import os
import argparse
import json
from dotenv import load_dotenv
from .together_rag import TogetherRAG, HAS_CHROMADB

# Load environment variables
load_dotenv()

def main():
    """Main function to run the Together RAG demo."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Together RAG Demo")
    parser.add_argument("--knowledge_dir", type=str, default="knowledgebase",
                        help="Directory containing knowledge base documents")
    parser.add_argument("--query", type=str, 
                        help="Query to test the RAG system with")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of documents to retrieve")
    parser.add_argument("--embedding_model", type=str, 
                        default="togethercomputer/m2-bert-80M-8k-retrieval",
                        help="Together embedding model to use")
    parser.add_argument("--llm_model", type=str, 
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="Together LLM model to use")
    parser.add_argument("--use_chroma", action="store_true", default=HAS_CHROMADB,
                        help="Use ChromaDB for vector storage")
    parser.add_argument("--chroma_dir", type=str, default="./chroma_db",
                        help="Directory to persist ChromaDB")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    print(f"Initializing Together RAG with embedding model: {args.embedding_model}")
    print(f"LLM model: {args.llm_model}")
    
    rag = TogetherRAG(
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        use_chroma=args.use_chroma,
        chroma_persist_directory=args.chroma_dir
    )
    
    # Load documents if the collection is empty
    if args.use_chroma and rag.use_chroma and rag.collection.count() == 0:
        print(f"Loading documents from {args.knowledge_dir}...")
        rag.load_markdown_files(args.knowledge_dir)
    elif not rag.use_chroma and not rag.documents:
        print(f"Loading documents from {args.knowledge_dir}...")
        rag.load_markdown_files(args.knowledge_dir)
    else:
        doc_count = rag.collection.count() if rag.use_chroma else len(rag.documents)
        print(f"Using existing documents ({doc_count} documents loaded)")
    
    # Process a single query if provided
    if args.query:
        process_query(rag, args.query, args.top_k)
    
    # Interactive mode
    if args.interactive:
        print("\n=== Together RAG Interactive Mode ===")
        print("Type 'exit' or 'quit' to end the session")
        
        while True:
            query = input("\nEnter your question: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            process_query(rag, query, args.top_k)

def process_query(rag, query, top_k):
    """Process a query and display the results."""
    print(f"\nQuery: {query}")
    print("Generating response...")
    
    result = rag.generate_response(query, top_k)
    
    print("\n=== Response ===")
    print(result["response"])
    
    print("\n=== Retrieved Documents ===")
    for i, doc in enumerate(result["retrieved_documents"]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc['metadata']['source']}")
        print(f"Chunk: {doc['metadata']['chunk_id'] + 1} of {doc['metadata']['total_chunks']}")
        if "similarity" in doc:
            print(f"Similarity: {doc['similarity']:.4f}")
        elif "distance" in doc and doc["distance"] is not None:
            print(f"Distance: {doc['distance']:.4f}")
        print("-" * 40)
        print(doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"])
        print("-" * 40)

if __name__ == "__main__":
    main() 