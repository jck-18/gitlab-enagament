#!/usr/bin/env python
"""
Demo script showing how to use the TogetherLLM class with the Llama-3.3-70B model.
"""

import os
import sys
from dotenv import load_dotenv
from .together_llm import TogetherLLM

def main():
    """Main function to demonstrate TogetherLLM functionality."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable not set.")
        print("Please set it in your .env file or pass it directly to the constructor.")
        sys.exit(1)
    
    # Initialize the LLM with the Llama-3.3-70B model
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    llm = TogetherLLM(api_key=api_key, model_name=model_name)
    
    # Example 1: Basic text generation
    prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in three sentences."
    print("\n🤖 Generating basic response...")
    response = llm.generate_response(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    
    # Example 2: Text generation with streaming
    prompt = "Write a short poem about artificial intelligence."
    print("\n🤖 Generating streamed response...")
    print(f"\nPrompt: {prompt}")
    print("Response: ", end="")
    for chunk in llm.stream_response(prompt):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Example 3: RAG with mock retrieved documents
    query = "What are the benefits of RAG systems?"
    retrieved_docs = [
        {
            "content": "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models. "
                       "It retrieves relevant documents from a knowledge base and incorporates them into the "
                       "context for the language model to generate more accurate and factual responses.",
            "metadata": {"source": "doc1.txt"}
        },
        {
            "content": "Benefits of RAG systems include: 1) Improved factuality as models can access external knowledge, "
                       "2) Better transparency since sources can be cited, 3) Reduced hallucinations as responses "
                       "are grounded in retrieved content, and 4) Cost efficiency by using smaller models with external knowledge.",
            "metadata": {"source": "doc2.txt"}
        }
    ]
    
    print("🤖 Generating RAG response...")
    print(f"\nQuery: {query}")
    response = llm.rag_generate(query, retrieved_docs)
    print(f"RAG Response: {response}")
    
    # Example 4: RAG with streaming
    print("\n🤖 Generating streamed RAG response...")
    print(f"\nQuery: {query}")
    print("RAG Response: ", end="")
    for chunk in llm.rag_generate(query, retrieved_docs, stream=True):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main() 