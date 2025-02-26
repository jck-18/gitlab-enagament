# GitLab Documentation RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) system for GitLab documentation. It processes markdown files from the knowledgebase directory, creates embeddings, and retrieves relevant information to answer user queries.

## Features

- Processes markdown files from the knowledgebase directory and its subdirectories
- Creates embeddings using Sentence Transformers
- Retrieves relevant documents based on cosine similarity
- Integrates with Reddit to process GitLab-related questions
- Generates responses based on retrieved documents
- Formats prompts for LLM integration
- **NEW**: Integration with Google Gemini API for response generation
- **NEW**: Support for different Gemini models (gemini-2.0-flash, etc.)
- **NEW**: Slack integration for notifications and interactive workflows

## Project Structure

```
.
├── knowledgebase/         # GitLab documentation in markdown format
├── src/                   # Source code
│   ├── rag_model.py       # RAG model implementation
│   ├── gitlab_rag_assistant.py  # Main script to run the RAG assistant
│   ├── reddit_retriever.py      # Script to retrieve Reddit posts
│   ├── llm_integration.py       # LLM integration with Google Gemini
│   ├── slack_webhook_server.py  # Flask server for Slack webhooks
│   ├── trigger_slack_workflow.py # Script to trigger Slack workflows
│   └── ...
├── results/               # Generated prompts and responses
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── SLACK_INTEGRATION.md   # Detailed Slack integration instructions
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the Google Gemini API key (optional, for LLM integration):

```bash
python src/gitlab_rag_assistant.py --setup-llm
```

4. Set up Slack integration (optional, see SLACK_INTEGRATION.md for details)

## Usage

### Process a specific query

```bash
python src/gitlab_rag_assistant.py --query "How do I set up GitLab CI/CD pipelines with Docker?"
```

### Process a query with LLM response generation

```bash
python src/gitlab_rag_assistant.py --query "How do I set up GitLab CI/CD pipelines with Docker?" --use-llm
```

### Process a query with a specific Gemini model

```bash
python src/gitlab_rag_assistant.py --query "How do I set up GitLab CI/CD pipelines with Docker?" --use-llm --llm-model "gemini-2.0-flash"
```

### Process Reddit posts

First, fetch new Reddit posts:

```bash
python src/reddit_retriever.py
```

Then, process the posts:

```bash
python src/gitlab_rag_assistant.py
```

### Use Slack Integration

Start the Slack webhook server:

```bash
python src/slack_webhook_server.py
```

Trigger a Slack workflow:

```bash
python src/trigger_slack_workflow.py --url "https://gitlab.com/docs/ci-cd" --title "How do I set up GitLab CI/CD pipelines with Docker?"
```

### Additional options

- `--knowledgebase`: Path to the knowledgebase directory (default: "knowledgebase")
- `--top-k`: Number of top documents to retrieve (default: 3)
- `--model`: Sentence transformer model to use (default: "all-MiniLM-L6-v2")
- `--use-llm`: Use LLM for response generation
- `--llm-model`: Gemini model to use (default: "gemini-2.0-flash")
- `--setup-llm`: Set up the LLM API key

## RAG Implementation Details

The RAG implementation consists of the following components:

1. **Document Processing**: Markdown files are loaded from the knowledgebase directory and chunked into smaller pieces.

2. **Embedding Creation**: The chunks are embedded using a Sentence Transformer model.

3. **Retrieval**: When a query is received, it is embedded and compared to the document embeddings using cosine similarity. The most relevant documents are retrieved.

4. **Response Generation**: A response is generated based on the retrieved documents. This can be done using a template or using the Google Gemini API.

5. **Prompt Formatting**: The retrieved documents and query are formatted into a prompt for an LLM.

## LLM Integration

The project integrates with the Google Gemini API for response generation. To use this feature:

1. Set up a Google API key at https://ai.google.dev/
2. Run `python src/gitlab_rag_assistant.py --setup-llm` to configure the API key
3. Use the `--use-llm` flag when running the assistant
4. Optionally, specify a Gemini model with `--llm-model` (default is "gemini-2.0-flash")

### Available Gemini Models

- `gemini-2.0-flash`: Balanced model for most use cases (default)
- `gemini-2.0-pro`: More powerful model for complex reasoning
- `gemini-1.5-flash`: Older version with good performance
- `gemini-1.5-pro`: Older version with more capabilities

## Slack Integration

The project integrates with Slack for notifications and interactive workflows. This allows you to:

1. Receive notifications in Slack when new GitLab-related Reddit posts are detected
2. Generate draft responses to these posts directly from Slack
3. Create custom workflows in Slack that trigger actions in the RAG system

For detailed setup instructions, see [SLACK_INTEGRATION.md](SLACK_INTEGRATION.md).

## Sample Prompt Construction

```
You are a helpful GitLab documentation assistant. 
Use the below references to answer the question accurately:

Reference 1: {chunk_text_1}
Reference 2: {chunk_text_2}
...

User Query: "{User's question: e.g. How do I set up GitLab CI/CD pipelines with Docker?}"

Instruction: Combine all references to create the best possible answer. If unsure, say so.
```

## Future Improvements

- Implement more sophisticated chunking strategies
- Add support for more document formats
- Integrate with additional LLM providers
- Implement a web interface for easier interaction
- Add support for document filtering based on metadata
- Enhance Slack integration with more interactive features
