# GitLab Documentation Knowledge Base Generator

This project creates a semantically organized knowledge base from GitLab documentation using advanced topic modeling and LLM-based content extraction. It automatically identifies relevant topics in the documentation, matches them with appropriate URLs, and extracts high-quality content while filtering out boilerplate and navigation elements.

## Overview

The GitLab Documentation Knowledge Base Generator uses a combination of:
- **Reddit integration** for community feedback and question monitoring
- **BERTopic** for advanced topic modeling
- **Semantic matching** to ensure relevance between topics and content
- **Together.ai's Llama 3.3 70B** for intelligent content extraction
- **Multi-stage filtering** to produce clean, useful documentation

## Problem Solved

This project addresses several challenges in creating useful knowledge bases from large documentation sites:

1. **Content Quality**: Most web crawlers capture excessive boilerplate, navigation elements, and irrelevant content
2. **Topic Organization**: Documentation is often organized by product structure rather than conceptual topics
3. **Relevance Matching**: Simple keyword matching often fails to capture semantic relationships
4. **Extraction Efficiency**: Processing large documentation sites can be resource-intensive

Our solution uses a combination of advanced NLP techniques and LLM-based extraction to create a high-quality, topic-organized knowledge base suitable for RAG (Retrieval-Augmented Generation) applications.

## Setup Instructions

### Prerequisites

- Python 3.8+ 
- Git
- Internet connection for API access

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/gitlab-mvp.git
cd gitlab-mvp
```

2. **Create and activate a virtual environment**

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Edit the `config/config.yaml` file to add your Together.ai API key:

```yaml
company:
  # Other settings...
  together_api_key: "YOUR_TOGETHER_API_KEY"
```

You can get a free API key by signing up at [Together.ai](https://www.together.ai/).

## Usage

The project consists of three main scripts:

1. **Reddit Monitoring**

```bash
python src/reddit_retriever.py
```

This script monitors specified subreddits for relevant posts about GitLab, saving them to a local database for analysis and response drafting.

2. **Topic Modeling**

```bash
python src/advanced_topic_modeling.py
```

This script analyzes GitLab documentation to identify key topics and their relationships.

3. **Knowledge Base Generation**

```bash
python src/docs_mapper.py
```

This script maps topics to relevant documentation URLs and extracts high-quality content to create a comprehensive knowledge base.

## Methodology

### 1. Reddit Community Monitoring

The project begins by monitoring Reddit for relevant discussions about GitLab:

- **Subreddit Monitoring**: Tracks specified subreddits for new posts
- **Keyword Filtering**: Identifies posts containing relevant keywords
- **Time-Based Filtering**: Focuses on recent posts (configurable timeframe)
- **Local Database Storage**: Saves relevant posts for analysis and response

The implementation includes:
- PRAW (Python Reddit API Wrapper) integration
- SQLite database for post storage
- Configurable subreddit and keyword lists
- Time-based filtering to focus on recent discussions

### 2. Topic Modeling with BERTopic

Based on community questions and documentation needs, the project uses BERTopic, an advanced topic modeling technique that leverages BERT embeddings and clustering to identify coherent topics within the documentation corpus. This approach provides more semantically meaningful topics compared to traditional methods like LDA.

Key steps:
- Document preprocessing and tokenization
- BERT-based embedding generation
- UMAP dimensionality reduction
- HDBSCAN clustering
- Topic representation extraction

The implementation includes:
- Timestamp-based versioning for topic models
- Customizable embedding models
- Adjustable clustering parameters
- Topic visualization capabilities

### 3. URL Matching and Content Relevance

A two-stage approach is used to match topics with relevant documentation:

1. **Initial URL-based filtering**:
   - Match topic keywords against URLs to create an initial candidate set
   - Limit the number of URLs per topic for efficiency

2. **Content-based semantic matching**:
   - Extract clean content from candidate URLs
   - Calculate TF-IDF vectors and cosine similarity with topic keywords
   - Apply a relevance threshold to ensure high-quality matches
   - Sort and select the most relevant URLs for each topic

The implementation includes:
- Asynchronous URL processing for improved performance
- Configurable similarity thresholds
- Error handling for failed requests
- Batch processing to manage server load

### 4. Intelligent Content Extraction

The project uses Together.ai's Llama 3.3 70B Instruct Turbo Free model to extract high-quality content:

- **Content filtering**: Removes navigation elements, headers, footers, and other boilerplate
- **Chunking**: Handles large documents by breaking them into manageable pieces
- **Instruction-guided extraction**: Uses specific instructions to focus on substantive technical content
- **Markdown formatting**: Ensures clean, structured output suitable for knowledge bases

The implementation includes:
- LLM-based extraction strategy using `crawl4ai`
- Custom prompt engineering for optimal content extraction
- Efficient token usage to minimize API costs
- Fallback mechanisms for handling extraction failures

### 5. Knowledge Base Generation

The final knowledge base is organized by topic, with each section containing:
- Topic header with representative keywords
- Extracted content from multiple relevant documentation pages
- Links to original documentation for reference

The implementation includes:
- Markdown formatting for easy readability
- Timestamp-based file naming for versioning
- Topic organization by relevance
- Source attribution for all content

## Project Structure

```
gitlab-mvp/
├── config/
│   └── config.yaml         # Configuration settings
├── results/                # Topic modeling results
│   ├── topic_info_*.csv    # Topic information
│   └── topic_representations_*.json  # Topic keyword representations
├── src/
│   ├── advanced_topic_modeling.py  # Topic modeling script
│   ├── docs_mapper.py      # Knowledge base generation script
│   └── reddit_retriever.py # Reddit monitoring script
├── reddit_posts.db         # SQLite database for Reddit posts
├── venv/                   # Virtual environment (not tracked in git)
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Implementation Details

### Reddit Retriever

The `reddit_retriever.py` script implements:

- Reddit API integration using PRAW
- Configurable subreddit monitoring
- Keyword-based post filtering
- Time-based filtering to focus on recent posts
- SQLite database storage for retrieved posts
- Preparation for response drafting

### Advanced Topic Modeling

The `advanced_topic_modeling.py` script implements:

- Custom document preprocessing for GitLab documentation
- BERTopic model configuration with optimized parameters
- Topic visualization and analysis
- Timestamp-based versioning for reproducibility
- Export of topic information and representations

### Documentation Mapping

The `docs_mapper.py` script implements:

- Two-stage URL matching (keyword-based and semantic)
- Asynchronous web crawling with error handling
- Content relevance analysis using TF-IDF and cosine similarity
- LLM-based content extraction using Together.ai's Llama 3.3 70B
- Knowledge base generation in markdown format

## Output

The knowledge base is saved as a Markdown file with the naming pattern `knowledge_base_YYYYMMDD_HHMMSS.md`, containing:
- Topics organized by sections
- Clean, formatted documentation content
- Links to original sources

## Future Improvements

Potential enhancements for future versions:

1. **Incremental updates**: Only process new or changed documentation
2. **Multi-source integration**: Combine documentation from multiple sources
3. **Custom embedding models**: Train domain-specific embeddings for improved topic modeling
4. **Interactive visualization**: Web interface for exploring the knowledge base
5. **Evaluation metrics**: Automated quality assessment of the generated knowledge base
6. **Automated Reddit responses**: Use the knowledge base to automatically draft responses to Reddit questions
7. **Sentiment analysis**: Track community sentiment about GitLab features and documentation

## License

[MIT License](LICENSE)

## Acknowledgments

- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
- [Together.ai](https://www.together.ai/) for providing the Llama 3.3 70B model
- [Crawl4AI](https://docs.crawl4ai.com/) for web crawling and content extraction
- [GitLab](https://docs.gitlab.com/) for their comprehensive documentation 