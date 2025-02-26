# Together AI RAG Implementation

This directory contains an implementation of a Retrieval-Augmented Generation (RAG) pipeline using the Together AI API. The implementation follows the steps provided in the [Together AI RAG documentation](https://docs.together.ai/docs/quickstart-retrieval-augmented-generation-rag).

## Files

- `together_llm.py`: Implementation of the Together AI LLM API client
- `together_embeddings.py`: Implementation of the Together AI embeddings API client
- `together_rag.py`: Main RAG implementation using Together AI
- `together_rag_demo.py`: Demo script to showcase the Together RAG pipeline

## Setup

1. Make sure you have the required dependencies installed:

```bash
pip install together numpy pandas tqdm python-dotenv
```

2. Optional: Install ChromaDB for vector storage (requires Microsoft Visual C++ 14.0 or greater):

```bash
pip install chromadb
```

If ChromaDB is not installed, the implementation will automatically fall back to in-memory vector storage.

3. Set up your Together AI API key:

   - Option 1: Set the `TOGETHER_API_KEY` environment variable
   - Option 2: Add your API key to `config/config.yaml` as `together_api_key: "your-api-key"`
   - Option 3: Pass your API key directly to the constructor

## Usage

### Basic Usage

```python
from together_rag import TogetherRAG

# Initialize the RAG system
rag = TogetherRAG(
    embedding_model_name="togethercomputer/m2-bert-80M-8k-retrieval",
    llm_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    use_chroma=True,  # Will fall back to in-memory if ChromaDB is not installed
    chroma_persist_directory="./chroma_db"
)

# Load documents
rag.load_markdown_files("knowledgebase")

# Generate a response
result = rag.generate_response("What is GitLab CI/CD?")
print(result["response"])
```

### Running the Demo

You can run the demo script to test the RAG pipeline:

```bash
python together_rag_demo.py --knowledge_dir knowledgebase --interactive
```

Command-line options:

- `--knowledge_dir`: Directory containing knowledge base documents (default: "knowledgebase")
- `--query`: Query to test the RAG system with
- `--top_k`: Number of documents to retrieve (default: 3)
- `--embedding_model`: Together embedding model to use (default: "togethercomputer/m2-bert-80M-8k-retrieval")
- `--llm_model`: Together LLM model to use (default: "mistralai/Mixtral-8x7B-Instruct-v0.1")
- `--use_chroma`: Use ChromaDB for vector storage (default: True if installed)
- `--chroma_dir`: Directory to persist ChromaDB (default: "./chroma_db")
- `--interactive`: Run in interactive mode

## How It Works

1. **Document Loading**: The system loads documents from a directory, processes them into chunks, and stores them.

2. **Embedding Generation**: Together AI's embedding models are used to generate embeddings for the documents.

3. **Vector Storage**: The embeddings are stored either in ChromaDB (persistent) or in memory.

4. **Retrieval**: When a query is received, the system generates an embedding for the query and finds the most similar documents.

5. **Response Generation**: The retrieved documents are used as context for the LLM to generate a response to the query.

## Available Models

### Embedding Models

- `togethercomputer/m2-bert-80M-8k-retrieval` (default)
- `togethercomputer/m2-bert-80M-32k-retrieval`
- Other models available on Together AI

### LLM Models

- `mistralai/Mixtral-8x7B-Instruct-v0.1` (default)
- `meta-llama/Llama-2-70b-chat-hf`
- `togethercomputer/llama-2-7b-chat`
- Other models available on Together AI

## Customization

You can customize the RAG pipeline by:

1. Changing the embedding model
2. Changing the LLM model
3. Adjusting the chunking parameters
4. Modifying the prompt template in `format_rag_prompt`
5. Implementing custom document loading logic

## Limitations

- The current implementation only supports markdown files
- The chunking strategy is simple and may not be optimal for all use cases
- The system does not implement advanced RAG techniques like re-ranking or query expansion
- ChromaDB is optional and requires Microsoft Visual C++ 14.0 or greater to install 