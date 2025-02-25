import xml.etree.ElementTree as ET
import yaml
import re
import requests
import json
import os
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import time
import random
import asyncio
import logging
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docs_mapper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(path='./config/config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_bertopic_results(results_dir='results'):
    """
    Load the most recent BERTopic modeling results.
    Returns topic representations and topic info.
    """
    try:
        # Find the most recent topic representation file
        representation_files = [f for f in os.listdir(results_dir) if f.startswith('topic_representations_')]
        if not representation_files:
            logger.error("No topic representation files found.")
            return None, None
        
        # Sort by timestamp (assuming format topic_representations_YYYYMMDD_HHMMSS.json)
        representation_files.sort(reverse=True)
        latest_rep_file = os.path.join(results_dir, representation_files[0])
        
        # Load topic representations
        with open(latest_rep_file, 'r') as f:
            topic_representations = json.load(f)
        
        # Find the most recent topic info file
        info_files = [f for f in os.listdir(results_dir) if f.startswith('topic_info_')]
        if not info_files:
            logger.warning("No topic info files found.")
            return topic_representations, None
        
        # Sort by timestamp
        info_files.sort(reverse=True)
        latest_info_file = os.path.join(results_dir, info_files[0])
        
        # Load topic info
        topic_info = pd.read_csv(latest_info_file)
        
        logger.info(f"Loaded BERTopic results from {latest_rep_file} and {latest_info_file}")
        return topic_representations, topic_info
        
    except Exception as e:
        logger.error(f"Error loading BERTopic results: {e}")
        return None, None

def parse_sitemap(sitemap_source):
    """
    Parse the XML sitemap.
    If sitemap_source is a URL (starts with 'http'), fetch the content first.
    Otherwise, assume it's a local file.
    Returns a list of URLs.
    """
    try:
        if sitemap_source.startswith("http"):
            logger.info(f"Fetching sitemap from URL: {sitemap_source}")
            response = requests.get(sitemap_source)
            response.raise_for_status()
            xml_content = response.content
            root = ET.fromstring(xml_content)
        else:
            logger.info(f"Loading sitemap from local file: {sitemap_source}")
            tree = ET.parse(sitemap_source)
            root = tree.getroot()

        # Assuming sitemap uses the <loc> tag for URLs.
        urls = [elem.text for elem in root.iter() if elem.tag.endswith("loc")]
        logger.info(f"Successfully parsed sitemap, found {len(urls)} URLs")
        return urls
    except Exception as e:
        logger.error(f"Error parsing sitemap: {e}")
        return []

def extract_topic_keywords(topic_representations, top_n=10):
    """
    Extract the top keywords for each topic from BERTopic results.
    Returns a dictionary mapping topic IDs to lists of keywords.
    """
    topic_keywords = {}
    for topic_id, data in topic_representations.items():
        # Convert topic_id from string to int if needed
        topic_id_int = int(topic_id) if isinstance(topic_id, str) else topic_id
        # Get the top N words for this topic
        keywords = data['words'][:top_n]
        topic_keywords[topic_id_int] = keywords
    
    logger.info(f"Extracted keywords for {len(topic_keywords)} topics")
    return topic_keywords

def match_topic_to_urls_initial(topic_keywords, urls, max_urls_per_topic=100):
    """
    Perform initial URL-based filtering to find potential matches between topics and URLs.
    This is just a first pass to reduce the number of URLs we need to analyze in depth.
    """
    initial_mapping = {}
    for topic_id, keywords in topic_keywords.items():
        matching_urls = []
        for url in urls:
            # Check if any keyword is in the URL
            if any(re.search(keyword, url, re.IGNORECASE) for keyword in keywords):
                matching_urls.append(url)
        
        # Limit the number of URLs per topic for efficiency
        if len(matching_urls) > max_urls_per_topic:
            logger.info(f"Topic {topic_id}: Limiting from {len(matching_urls)} to {max_urls_per_topic} URLs for content analysis")
            matching_urls = matching_urls[:max_urls_per_topic]
        
        initial_mapping[topic_id] = matching_urls
        logger.info(f"Topic {topic_id}: Found {len(matching_urls)} initial URL matches")
    
    return initial_mapping

async def analyze_content_relevance(topic_id, keywords, urls, together_api_key):
    """
    Analyze the content of URLs to determine their relevance to a topic.
    Uses Crawl4AI to extract clean content and then performs semantic matching.
    """
    if not urls:
        logger.warning(f"No URLs to analyze for Topic {topic_id}")
        return []
    
    logger.info(f"Analyzing content relevance for Topic {topic_id} with {len(urls)} URLs")
    
    # Create a query string from the topic keywords
    topic_query = ' '.join(keywords)
    
    # Configure the crawler to extract clean content without boilerplate
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        word_count_threshold=50,  # Ensure we get meaningful content
        content_filter={
            "prune_elements": ["nav", "header", "footer", "aside", ".sidebar", ".navigation", ".menu"],
            "query_relevance": {
                "query": topic_query,
                "min_score": 0.1  # Minimum BM25 relevance score
            }
        }
    )
    
    relevant_urls = []
    
    # Process URLs in batches to avoid overwhelming the server
    batch_size = 5
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size} for Topic {topic_id}")
        
        async with AsyncWebCrawler(api_key=together_api_key) as crawler:
            # Process batch in parallel
            tasks = [crawler.arun(url=url, config=config) for url in batch_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, result in zip(batch_urls, results):
                if isinstance(result, Exception):
                    logger.error(f"Error crawling {url}: {result}")
                    continue
                
                # Check if we got meaningful content
                if result.markdown and len(result.markdown.strip()) > 200:
                    # Calculate relevance score using TF-IDF and cosine similarity
                    vectorizer = TfidfVectorizer(stop_words='english')
                    try:
                        tfidf_matrix = vectorizer.fit_transform([topic_query, result.markdown])
                        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        
                        logger.info(f"URL: {url} - Similarity score: {similarity:.4f}")
                        
                        if similarity > 0.15:  # Higher threshold for final selection
                            relevant_urls.append((url, similarity))
                            logger.info(f"Added {url} to relevant URLs for Topic {topic_id} (score: {similarity:.4f})")
                    except Exception as e:
                        logger.error(f"Error calculating similarity for {url}: {e}")
                else:
                    logger.warning(f"Insufficient content from {url} (length: {len(result.markdown.strip()) if result.markdown else 0})")
        
        # Add a small delay between batches
        await asyncio.sleep(2)
    
    # Sort by relevance and take top results
    relevant_urls.sort(key=lambda x: x[1], reverse=True)
    top_n = min(10, len(relevant_urls))  # Limit to top 10 most relevant URLs
    
    final_urls = [url for url, _ in relevant_urls[:top_n]]
    logger.info(f"Final selection for Topic {topic_id}: {len(final_urls)} URLs")
    
    return final_urls

async def match_topic_to_urls_semantic(topic_keywords, urls, together_api_key, max_urls_per_topic=100):
    """
    Match topics to URLs using a two-step process:
    1. Initial filtering based on URL keywords
    2. Content-based semantic matching using Crawl4AI
    """
    # Step 1: Initial URL-based filtering
    logger.info("Starting initial URL-based filtering")
    initial_mapping = match_topic_to_urls_initial(topic_keywords, urls, max_urls_per_topic)
    
    # Step 2: Content-based semantic matching
    logger.info("Starting content-based semantic matching")
    final_mapping = {}
    
    for topic_id, keywords in topic_keywords.items():
        if not initial_mapping[topic_id]:
            logger.warning(f"No initial URL matches for Topic {topic_id}, skipping content analysis")
            final_mapping[topic_id] = []
            continue
        
        relevant_urls = await analyze_content_relevance(
            topic_id, 
            keywords, 
            initial_mapping[topic_id], 
            together_api_key
        )
        
        final_mapping[topic_id] = relevant_urls
    
    return final_mapping

async def test_llm_extraction(url, together_api_key):
    """
    Test LLM extraction on a single URL to verify it's working correctly.
    Returns the extracted content and any errors encountered.
    """
    logger.info(f"Testing LLM extraction on URL: {url}")
    
    try:
        # Create an instruction for the LLM to extract relevant content
        instruction = """
        Extract the main technical documentation content from this page.
        Ignore navigation elements, headers, footers, and other boilerplate.
        Focus on extracting substantive information that would be useful for a knowledge base.
        Format the output as clean markdown with proper headings and structure.
        """
        
        # Configure LLM extraction strategy with Together.ai's Llama 3.3
        llm_strategy = LLMExtractionStrategy(
            provider="together",  # Just specify "together" as the provider
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Specify the model separately
            api_token=together_api_key,
            extraction_type="block",
            instruction=instruction,
            input_format="fit_markdown",
            chunk_token_threshold=4000,
            apply_chunking=True,
            extra_args={
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1,
                "stop": ["<|eot_id|>", "<|eom_id|>"]
            }
        )
        
        # Configure the crawler
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=30000,
            word_count_threshold=50,
            extraction_strategy=llm_strategy,
            content_filter={
                "prune_elements": ["nav", "header", "footer", "aside", ".sidebar", ".navigation", ".menu"]
            }
        )
        
        # Initialize AsyncWebCrawler and crawl the page
        async with AsyncWebCrawler(api_key=together_api_key) as crawler:
            result = await crawler.arun(url=url, config=config)
            
            if result.extracted_content:
                logger.info(f"LLM extraction successful for {url}")
                return {
                    "success": True,
                    "content": result.extracted_content,
                    "content_length": len(result.extracted_content),
                    "message": "LLM extraction successful"
                }
            else:
                logger.warning(f"LLM extraction returned no content for {url}")
                return {
                    "success": False,
                    "content": None,
                    "message": "LLM extraction returned no content"
                }
                
    except Exception as e:
        logger.error(f"Error testing LLM extraction for {url}: {e}")
        return {
            "success": False,
            "content": None,
            "message": f"Error: {str(e)}"
        }

def verify_content_quality(content):
    """
    Verify that the extracted content is of good quality.
    Returns a tuple of (is_good_quality, reason).
    """
    if not content:
        return False, "Content is empty"
    
    # Check content length
    if len(content) < 200:
        return False, "Content is too short"
    
    # Check for error messages in the content
    error_patterns = [
        "Error crawling",
        "Could not extract",
        "litellm.BadRequestError",
        "LLM Provider NOT provided"
    ]
    
    for pattern in error_patterns:
        if pattern in content:
            return False, f"Content contains error message: {pattern}"
    
    # Check for meaningful structure (headers, paragraphs)
    if not re.search(r'#+ ', content) and not re.search(r'\n\n', content):
        return False, "Content lacks structure (no headers or paragraphs)"
    
    return True, "Content appears to be of good quality"

async def crawl_page_with_llm(url, together_api_key, topic_keywords):
    """
    Crawl a page and extract its content using Together.ai's Llama 3.3 70B Instruct Turbo Free model.
    """
    try:
        logger.info(f"Crawling {url} with LLM extraction...")
        
        # Create an instruction for the LLM to extract relevant content
        instruction = """
        Extract the main technical documentation content from this page.
        Ignore navigation elements, headers, footers, and other boilerplate.
        Focus on extracting substantive information that would be useful for a knowledge base.
        Format the output as clean markdown with proper headings and structure.
        """
        
        # Configure LLM extraction strategy with Together.ai's Llama 3.3
        llm_strategy = LLMExtractionStrategy(
            provider="together",  # Just specify "together" as the provider
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Specify the model separately
            api_token=together_api_key,
            extraction_type="block",
            instruction=instruction,
            input_format="fit_markdown",
            chunk_token_threshold=4000,
            apply_chunking=True,
            extra_args={
                "temperature": 0.1,  # Lower temperature for more focused extraction
                "max_tokens": 4096,  # Increased from 2048 to capture more content
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1,
                "stop": ["<|eot_id|>", "<|eom_id|>"]
            }
        )
        
        # Configure the crawler
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=30000,
            word_count_threshold=50,
            extraction_strategy=llm_strategy,
            content_filter={
                "prune_elements": ["nav", "header", "footer", "aside", ".sidebar", ".navigation", ".menu"]
            }
        )
        
        # Initialize AsyncWebCrawler and crawl the page
        async with AsyncWebCrawler(api_key=together_api_key) as crawler:
            result = await crawler.arun(url=url, config=config)
            
            content = None
            
            if result.extracted_content:
                # The LLM has already extracted and formatted the content
                content = result.extracted_content
                logger.info(f"LLM extraction successful for {url} - Content length: {len(content)}")
            elif result.markdown:
                # Use markdown if LLM extraction failed
                content = f"## {result.metadata.get('title', 'No title found')}\n\n{result.markdown}"
                logger.warning(f"Falling back to markdown for {url} - Content length: {len(content)}")
            else:
                # Fallback to basic text
                error_msg = f"## Error extracting content from {url}\n\nCould not extract meaningful content."
                content = error_msg
                logger.error(f"Failed to extract content from {url}")
            
            # Verify content quality
            is_good_quality, reason = verify_content_quality(content)
            if not is_good_quality:
                logger.warning(f"Low quality content from {url}: {reason}")
                # If content is low quality, we might want to try a different approach
                # For now, we'll just log the issue and continue
            
            # Add link to original document
            content += f"\n\n[Read full documentation]({url})"
            
            return content
            
    except Exception as e:
        logger.error(f"Error crawling {url}: {e}")
        return f"## Error crawling {url}\n\n{str(e)}\n\n[Link]({url})"

async def build_knowledge_base(mapping, together_api_key, topic_representations):
    """
    For each topic and its list of URLs, crawl each URL and aggregate the content.
    Uses LLM-based extraction for higher quality content.
    Returns a dictionary mapping topics to aggregated documentation.
    """
    knowledge_base = {}
    
    for topic_id, urls in mapping.items():
        if not urls:
            logger.warning(f"No URLs for Topic {topic_id}, skipping")
            continue
            
        # Get the topic keywords for reference
        topic_words = topic_representations.get(str(topic_id), {}).get('words', [])
        topic_header = f"Topic {topic_id}: {', '.join(topic_words[:5])}"
        
        logger.info(f"Building knowledge base for {topic_header} with {len(urls)} URLs")
        
        docs_content = [f"# {topic_header}\n\n"]
        successful_extractions = 0
        
        # Process all URLs for this topic
        for i, url in enumerate(urls):
            logger.info(f"  Processing URL {i+1}/{len(urls)}: {url}")
            content = await crawl_page_with_llm(url, together_api_key, topic_words)
            
            # Check if extraction was successful (not just an error message)
            is_good_quality, reason = verify_content_quality(content)
            if is_good_quality:
                successful_extractions += 1
            
            docs_content.append(content)
            docs_content.append("\n\n---\n\n")  # Add separator between documents
        
        # Log extraction success rate
        success_rate = (successful_extractions / len(urls)) * 100 if urls else 0
        logger.info(f"Topic {topic_id}: Extraction success rate: {success_rate:.2f}% ({successful_extractions}/{len(urls)})")
        
        # Aggregate all contents for this topic
        knowledge_base[topic_id] = "\n\n".join(docs_content)
    
    return knowledge_base

async def main_async():
    config = load_config()
    company_cfg = config.get('company', {})
    sitemap_source = company_cfg.get('docs_sitemap', "https://docs.gitlab.com/sitemap.xml")
    together_api_key = company_cfg.get('together_api_key', "")
    docs_url = company_cfg.get('docs_url', "https://docs.gitlab.com")

    if not together_api_key:
        logger.error("Error: Together API key not found in config. Please add 'together_api_key' to your config file.")
        return

    # Load BERTopic results
    topic_representations, topic_info = load_bertopic_results()
    if not topic_representations:
        logger.error("No BERTopic results found. Please run advanced_topic_modeling.py first.")
        return

    # Extract keywords for each topic
    topic_keywords = extract_topic_keywords(topic_representations)
    logger.info(f"Extracted keywords for {len(topic_keywords)} topics")

    # Parse sitemap to get documentation URLs
    urls = parse_sitemap(sitemap_source)
    logger.info(f"Parsed {len(urls)} URLs from sitemap.")
    
    # Test LLM extraction on a single URL first
    if urls:
        logger.info("Testing LLM extraction on a single URL before proceeding...")
        test_url = urls[0]  # Use the first URL for testing
        test_result = await test_llm_extraction(test_url, together_api_key)
        
        if not test_result["success"]:
            logger.error(f"LLM extraction test failed: {test_result['message']}")
            logger.error("Please check your Together API key and model configuration.")
            logger.error("Aborting knowledge base generation.")
            return
        else:
            logger.info(f"LLM extraction test successful! Content length: {test_result['content_length']}")
    else:
        logger.error("No URLs found in sitemap. Cannot proceed with knowledge base generation.")
        return

    # Match topics to URLs using semantic matching
    logger.info("Performing semantic matching of topics to URLs...")
    mapping = await match_topic_to_urls_semantic(topic_keywords, urls, together_api_key)
    
    # Print summary of mapping
    logger.info("\nTopic to URL mapping summary (after semantic matching):")
    for topic_id, urls in mapping.items():
        keywords = topic_keywords.get(topic_id, [])
        logger.info(f"Topic {topic_id} ({', '.join(keywords[:3])}...): {len(urls)} matching URLs")
        if urls:
            logger.info(f"  Top 3 URLs:")
            for url in urls[:3]:
                logger.info(f"  - {url}")

    # Build knowledge base by crawling matched URLs
    knowledge_base = await build_knowledge_base(mapping, together_api_key, topic_representations)
    
    # Save the knowledge base
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"knowledge_base_{timestamp}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        for topic_id, content in knowledge_base.items():
            f.write(f"{content}\n\n---\n\n")
    
    # Calculate statistics for the knowledge base
    total_topics = len(knowledge_base)
    total_content_length = sum(len(content) for content in knowledge_base.values())
    avg_content_length = total_content_length / total_topics if total_topics > 0 else 0
    
    logger.info(f"Knowledge base built and saved as '{output_file}'.")
    logger.info(f"Contains information for {total_topics} topics from BERTopic modeling.")
    logger.info(f"Total content length: {total_content_length} characters")
    logger.info(f"Average content per topic: {avg_content_length:.2f} characters")

def main():
    start_time = time.time()
    logger.info("Starting docs_mapper.py")
    
    try:
        asyncio.run(main_async())
        elapsed_time = time.time() - start_time
        logger.info(f"docs_mapper.py completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        logger.error(f"docs_mapper.py failed after {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
