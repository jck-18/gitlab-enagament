import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini LLM.
        
        Args:
            api_key: Google API key for Gemini. If None, will try to load from environment variable.
            model_name: The Gemini model to use. Default is "gemini-2.0-flash".
        """
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass an API key to the constructor or "
                "set the GOOGLE_API_KEY environment variable."
            )
        
        # Initialize the Gemini client
        genai.configure(api_key=self.api_key)
        
        # Set the model name
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"Initialized Gemini LLM with model: {self.model_name}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Gemini model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated response
        """
        try:
            response = self.model.generate_content(contents=prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def format_rag_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Format a RAG prompt for the Gemini model.
        
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
    
    def rag_generate(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate a response based on retrieved documents.
        
        Args:
            query: The user query
            retrieved_docs: The retrieved documents
            
        Returns:
            Generated response
        """
        prompt = self.format_rag_prompt(query, retrieved_docs)
        return self.generate_response(prompt)


def setup_gemini_api():
    """
    Guide the user through setting up the Gemini API key.
    """
    print("Setting up Google Gemini API...")
    
    # Check if API key is already set
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        print("Google API key found in environment variables.")
        return
    
    # Guide the user to set up the API key
    print("\nTo use the Google Gemini API, you need to set up an API key:")
    print("1. Go to https://ai.google.dev/")
    print("2. Create an API key")
    print("3. Create a .env file in the project root directory")
    print("4. Add the following line to the .env file:")
    print("   GOOGLE_API_KEY=your_api_key_here")
    print("\nAlternatively, you can set the GOOGLE_API_KEY environment variable.")
    
    # Ask the user if they want to enter the API key now
    api_key_input = input("\nDo you want to enter your API key now? (y/n): ")
    
    if api_key_input.lower() == 'y':
        api_key = input("Enter your Google API key: ")
        
        # Create .env file
        with open(".env", "w") as f:
            f.write(f"GOOGLE_API_KEY={api_key}")
        
        print("API key saved to .env file.")
        
        # Load the new environment variable
        load_dotenv(override=True)
    else:
        print("Please set up the API key manually before using the Gemini integration.")


if __name__ == "__main__":
    # Example usage
    setup_gemini_api()
    
    try:
        llm = GeminiLLM()
        
        # Test the model
        response = llm.generate_response("What is GitLab?")
        print("\nTest response:")
        print(response)
    except ValueError as e:
        print(e) 