# GitLab Documentation RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) system for GitLab documentation using Together AI's API. It processes markdown files from the knowledgebase directory, creates embeddings, and retrieves relevant information to answer user queries.

I EDITED THE FILE STRUCTURE AT THE LAST MINUITE AND HENCE A LOT OF FILES WERE NOT RUNNING DURING THE VIDEO PRESENTATION, TILL BEFORE THE ORGANIZE DIRECTORY STRUCTURE COMMIT ALL THE FIELS WERE PROPERLY GIVING OUTPUTS

## Features

- Processes markdown files from the knowledgebase directory and its subdirectories
- Creates embeddings using Together AI's embedding models
- Retrieves relevant documents based on semantic similarity
- Integrates with Reddit to process GitLab-related questions
- Generates responses using Together AI's LLM models
- Formats prompts for LLM integration
- Slack integration for notifications and interactive workflows
- ChromaDB integration for persistent vector storage
- Interactive mode for real-time querying
- Support for multiple Together AI models

## Project Structure

```
.
├── knowledgebase/         # GitLab documentation in markdown format
├── src/                   # Source code
│   ├── core/             # Core RAG implementation
│   │   ├── together_rag.py       # Main RAG implementation
│   │   └── test_rag.py          # Testing utilities
│   ├── models/           # Model implementations
│   │   ├── together_llm.py       # LLM integration
│   │   └── together_embeddings.py # Embeddings integration
│   ├── integrations/     # External integrations
│   │   ├── slack/              # Slack integration
│   │   └── reddit/             # Reddit integration
│   └── utils/            # Utility functions
├── config/               # Configuration files
├── results/              # Generated prompts and responses
├── chroma_db/            # ChromaDB persistence directory
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── SLACK_INTEGRATION.md  # Detailed Slack integration instructions
└── together_rag_README.md # Together AI RAG implementation details
```

## Documentation

- [Main README](README.md) - Overview and general setup
- [Together RAG Documentation](together_rag_README.md) - Details about the Together AI RAG implementation
- [Slack Integration Guide](SLACK_INTEGRATION.md) - Instructions for setting up Slack integration

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the Together AI API key:
   - Set the `TOGETHER_API_KEY` environment variable, or
   - Add it to `config/config.yaml`, or
   - Pass it directly to the constructor

4. (Optional) Install ChromaDB for persistent vector storage:
```bash
pip install chromadb
```

5. (Optional) Set up Slack integration (see [SLACK_INTEGRATION.md](SLACK_INTEGRATION.md) for details)

## Security and Environment Variables

To keep your API keys and sensitive information secure:

1. Copy `.env.example` to `.env` and fill in your actual API keys and secrets:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your actual credentials:
```
TOGETHER_API_KEY=your_actual_api_key
SLACK_WEBHOOK_URL=your_actual_webhook_url
SLACK_WORKFLOW_WEBHOOK_URL=your_actual_workflow_url
SLACK_SIGNING_SECRET=your_actual_signing_secret
```

3. Copy `config/config.yaml.example` to `config/config.yaml` and customize it:
```bash
cp config/config.yaml.example config/config.yaml
```

4. **IMPORTANT**: Never commit your `.env` file or `config/config.yaml` to version control. They are already added to `.gitignore`.

5. When using ngrok for development, be aware that the generated URLs provide public access to your local server. Never share these URLs publicly or commit them to version control.

## Usage

### Basic Query Processing

```bash
python src/core/test_rag.py --query "How do I set up GitLab CI/CD pipelines with Docker?"
```

### Interactive Mode

```bash
python src/core/together_rag_demo.py --knowledge_dir knowledgebase --interactive
```

### Process Reddit Posts

```bash
python src/integrations/reddit/reddit_retriever.py
```

### Use Slack Integration

```bash
python src/integrations/slack/slack_webhook_server.py
```

### Additional Options

- `--knowledge_dir`: Path to the knowledgebase directory (default: "knowledgebase")
- `--top_k`: Number of documents to retrieve (default: 3)
- `--embedding_model`: Together AI embedding model to use (default: "togethercomputer/m2-bert-80M-8k-retrieval")
- `--llm_model`: Together AI LLM model to use (default: "mistralai/Mixtral-8x7B-Instruct-v0.1")
- `--use_chroma`: Use ChromaDB for vector storage
- `--chroma_dir`: Directory for ChromaDB persistence

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

## RAG Implementation Details

The RAG implementation consists of the following components:

1. **Document Processing**: Markdown files are loaded and chunked into smaller pieces.
2. **Embedding Creation**: Chunks are embedded using Together AI's embedding models.
3. **Vector Storage**: Embeddings are stored in ChromaDB (if installed) or in memory.
4. **Retrieval**: Queries are embedded and compared to document embeddings using semantic similarity.
5. **Response Generation**: Responses are generated using Together AI's LLM models.

For detailed implementation information, see [together_rag_README.md](together_rag_README.md).

## Slack Integration

The project includes comprehensive Slack integration features:
- Receive notifications for new GitLab-related Reddit posts
- Generate draft responses directly from Slack
- Run RAG analysis on posts with interactive buttons
- Create custom workflows for automation

For setup instructions and details, see [SLACK_INTEGRATION.md](SLACK_INTEGRATION.md).

## Future Improvements

- Implement advanced RAG techniques (re-ranking, query expansion)
- Add support for more document formats
- Enhance chunking strategies
- Add web interface for easier interaction
- Implement document filtering based on metadata
- Add support for more LLM providers
- Enhance Slack integration with more interactive features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
