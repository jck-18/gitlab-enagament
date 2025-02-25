import os
import sys
import argparse
from rag_model import GitLabRAG
from reddit_retriever import retrieve_reddit_posts

# Import the LLM setup function if available
try:
    from llm_integration import setup_gemini_api
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

def main():
    """
    Main function to run the GitLab RAG assistant.
    """
    parser = argparse.ArgumentParser(description="GitLab RAG Assistant")
    parser.add_argument("--knowledgebase", type=str, default="knowledgebase", 
                        help="Path to the knowledgebase directory")
    parser.add_argument("--fetch-reddit", action="store_true", 
                        help="Fetch new Reddit posts before processing")
    parser.add_argument("--query", type=str, 
                        help="Process a specific query instead of Reddit posts")
    parser.add_argument("--top-k", type=int, default=3, 
                        help="Number of top documents to retrieve")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="Sentence transformer model to use")
    parser.add_argument("--use-llm", action="store_true", 
                        help="Use LLM for response generation")
    parser.add_argument("--llm-model", type=str, default="gemini-2.0-flash",
                        help="Gemini model to use (default: gemini-2.0-flash)")
    parser.add_argument("--setup-llm", action="store_true", 
                        help="Set up the LLM API key")
    
    args = parser.parse_args()
    
    # Set up the LLM API key if requested
    if args.setup_llm and HAS_GEMINI:
        setup_gemini_api()
        return
    
    # Initialize the RAG model
    print("Initializing GitLab RAG model...")
    rag = GitLabRAG(model_name=args.model, use_llm=args.use_llm, llm_model_name=args.llm_model)
    
    # Load markdown files from the knowledgebase
    rag.load_markdown_files(args.knowledgebase)
    
    # Create embeddings
    rag.create_embeddings()
    
    # Fetch new Reddit posts if requested
    if args.fetch_reddit:
        print("Fetching new Reddit posts...")
        retrieve_reddit_posts()
    
    # Process a specific query if provided
    if args.query:
        process_query(rag, args.query, args.top_k)
    else:
        # Process Reddit posts
        process_reddit_posts(rag, args.top_k)

def process_query(rag, query, top_k):
    """
    Process a specific query and print the results.
    
    Args:
        rag: The GitLabRAG instance
        query: The query to process
        top_k: Number of top documents to retrieve
    """
    print(f"\nProcessing query: {query}")
    
    # Retrieve relevant documents
    results = rag.retrieve(query, top_k=top_k)
    
    # Print results
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
    
    # Format the response as a prompt for an LLM
    prompt = format_prompt(query, results)
    print("\nFormatted prompt for LLM:")
    print(prompt)

def process_reddit_posts(rag, top_k):
    """
    Process Reddit posts and save responses to the database.
    
    Args:
        rag: The GitLabRAG instance
        top_k: Number of top documents to retrieve
    """
    # Get Reddit posts
    posts = rag.get_reddit_posts()
    
    if not posts:
        print("No Reddit posts found. Run with --fetch-reddit to fetch new posts.")
        return
    
    print(f"Found {len(posts)} Reddit posts to process.")
    
    for i, post in enumerate(posts):
        print(f"\nProcessing post {i+1}/{len(posts)}: {post['title']}")
        
        # Combine title and content for the query
        query = f"{post['title']} {post['content']}"
        
        # Retrieve relevant documents
        results = rag.retrieve(query, top_k=top_k)
        
        # Generate a response
        response = rag.generate_response(query, results)
        
        # Save the response to the database
        rag.save_to_db(post['id'], response)
        
        # Format the prompt for an LLM
        prompt = format_prompt(query, results)
        
        # Print a preview
        print(f"Generated response for: {post['title']}")
        print(f"URL: {post['url']}")
        
        # Save the prompt to a file
        save_prompt_to_file(post['id'], prompt)

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

def save_prompt_to_file(post_id, prompt, output_dir="results"):
    """
    Save a prompt to a file.
    
    Args:
        post_id: The ID of the Reddit post
        prompt: The formatted prompt
        output_dir: The directory to save the file to
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prompt to a file
    with open(f"{output_dir}/{post_id}_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    
    print(f"Saved prompt to {output_dir}/{post_id}_prompt.txt")

if __name__ == "__main__":
    main() 