import os
import json
from typing import List, Dict, Optional, Union, Any, Generator, cast
from dotenv import load_dotenv
from together import Together

# Load environment variables from .env file
load_dotenv()

class TogetherLLM:
    """
    A class for interacting with Together AI's LLM API using the official SDK.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """
        Initialize the Together LLM.
        
        Args:
            api_key: Together API key. If None, will try to load from environment variable.
            model_name: The model to use. Default is "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free".
        """
        # Get API key from environment variable or config if not provided
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
                
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass an API key to the constructor, "
                "set the TOGETHER_API_KEY environment variable, or add it to config/config.yaml."
            )
        
        # Set the model name
        self.model_name = model_name
        
        # Initialize the Together client
        self.client = Together(api_key=self.api_key)
        
        print(f"Initialized Together LLM with model: {self.model_name}")
    
    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024, stream: bool = False) -> Any:
        """
        Generate a response using the Together API.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: The maximum number of tokens to generate.
            stream: Whether to stream the response.
            
        Returns:
            The generated response as a string, or a stream of response chunks if stream=True
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Cast to Any to avoid type checking issues
            response: Any = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stream=stream
            )
            
            if stream:
                return response
            else:
                # For non-streaming, extract the content from the response
                try:
                    if response and hasattr(response, 'choices') and response.choices:
                        return response.choices[0].message.content
                    return "No response generated"
                except (AttributeError, IndexError):
                    return "No response generated"
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def stream_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> Generator[str, None, None]:
        """
        Stream a response from the model.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of the generated text
        """
        try:
            # Cast to Any to avoid type checking issues
            response_stream: Any = self.generate_response(prompt, temperature, max_tokens, stream=True)
            
            # The response_stream is an iterator of chunks
            for chunk in response_stream:
                try:
                    # Access delta content if it exists
                    content = None
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = getattr(chunk.choices[0], 'delta', None)
                        if delta and hasattr(delta, 'content'):
                            content = delta.content
                    
                    if content:
                        yield content
                except (AttributeError, IndexError):
                    continue
        except Exception as e:
            yield f"Error streaming response: {str(e)}"
    
    def format_rag_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Format a prompt for RAG using retrieved documents.
        
        Args:
            query: The user's query
            retrieved_docs: List of retrieved documents with their content and metadata
            
        Returns:
            A formatted prompt for the LLM
        """
        context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""You are a helpful assistant that provides accurate information based on the given context.
        
Context:
{context}

User Question: {query}

Please provide a comprehensive answer to the question based only on the information in the context. 
If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer this question."
"""
        return prompt
    
    def rag_generate(self, query: str, retrieved_docs: List[Dict], stream: bool = False) -> Any:
        """
        Generate a response using RAG.
        
        Args:
            query: The user's query
            retrieved_docs: List of retrieved documents with their content and metadata
            stream: Whether to stream the response
            
        Returns:
            The generated response or a stream of response chunks
        """
        prompt = self.format_rag_prompt(query, retrieved_docs)
        
        if stream:
            return self.stream_response(prompt)
        else:
            return self.generate_response(prompt) 