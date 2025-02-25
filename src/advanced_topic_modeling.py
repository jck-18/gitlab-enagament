import sqlite3
import yaml
import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def load_config(path='./config/config.yaml'):
    """Load configuration from YAML file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def fetch_posts():
    """Fetch posts from SQLite database with timestamps if available."""
    connection = sqlite3.connect('reddit_posts.db')
    cursor = connection.cursor()
    
    # Check if created_utc column exists
    cursor.execute("PRAGMA table_info(posts)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'created_utc' in columns:
        cursor.execute("SELECT content, created_utc FROM posts WHERE content IS NOT NULL")
        posts = cursor.fetchall()
        connection.close()
        return [(post[0], post[1]) for post in posts if post[0].strip() != ""]
    else:
        # If created_utc doesn't exist, use current timestamp for all posts
        cursor.execute("SELECT content FROM posts WHERE content IS NOT NULL")
        posts = cursor.fetchall()
        connection.close()
        current_time = int(time.time())
        return [(post[0], current_time) for post in posts if post[0].strip() != ""]

def create_topic_model():
    """Create and configure a BERTopic model."""
    # Initialize sentence transformer for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Configure vectorizer for topic representation
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2)
    )
    
    # Create BERTopic model with all components
    topic_model = BERTopic(
        embedding_model=embedding_model,  # Use the embedding model directly
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=True
    )
    
    return topic_model

def perform_topic_modeling(docs, timestamps=None):
    """Perform topic modeling on documents with optional timestamps."""
    print(f"Starting topic modeling with {len(docs)} documents...")
    
    # Create the topic model
    topic_model = create_topic_model()
    
    # Fit the model to the documents
    if timestamps:
        print("Using temporal topic modeling...")
        # For temporal analysis, we first fit the model normally
        topics, probs = topic_model.fit_transform(docs)
    else:
        # Standard topic modeling
        topics, probs = topic_model.fit_transform(docs)
    
    print(f"Found {len(topic_model.get_topic_info()) - 1} topics")
    return topic_model, topics, probs

def analyze_topics(topic_model, docs, timestamps=None):
    """Analyze topics and generate visualizations and insights."""
    # Get basic topic information
    topic_info = topic_model.get_topic_info()
    print(f"Topic modeling complete. Found {len(topic_info) - 1} topics.")
    
    # Extract topic representations
    topic_representations = {}
    for topic in topic_info['Topic']:
        if topic != -1:  # Skip outlier topic
            topic_words = topic_model.get_topic(topic)
            if topic_words:  # Check if topic has words
                words, scores = zip(*topic_words)
                topic_representations[topic] = {
                    'words': words[:10],
                    'scores': [float(score) for score in scores[:10]]  # Convert to float for JSON serialization
                }
    
    # Generate visualizations
    try:
        topic_model.visualize_topics()
    except Exception as e:
        print(f"Warning: Could not generate topic visualization: {str(e)}")
    
    try:
        topic_model.visualize_hierarchy()
    except Exception as e:
        print(f"Warning: Could not generate hierarchy visualization: {str(e)}")
    
    # Perform temporal analysis if timestamps are provided
    topics_over_time = None
    if timestamps:
        try:
            # Convert timestamps to the right format if needed
            if isinstance(timestamps[0], (int, float)):
                # Convert UNIX timestamps to datetime for better visualization
                timestamps_list = timestamps
            else:
                timestamps_list = list(timestamps)
            
            # Create temporal topic analysis
            topics_over_time = topic_model.topics_over_time(
                docs,
                timestamps_list,
                global_tuning=True
            )
            
            # Visualize topics over time
            try:
                topic_model.visualize_topics_over_time(topics_over_time)
            except Exception as e:
                print(f"Warning: Could not visualize topics over time: {str(e)}")
                
        except Exception as e:
            print(f"Warning: Could not perform temporal analysis: {str(e)}")
    
    # Return analysis results
    return {
        'topic_info': topic_info,
        'representations': topic_representations,
        'topics_over_time': topics_over_time
    }

def save_results(analysis_results, output_dir='results'):
    """Save analysis results to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save topic info
    topic_info_df = analysis_results['topic_info']
    topic_info_path = f'{output_dir}/topic_info_{timestamp}.csv'
    topic_info_df.to_csv(topic_info_path)
    print(f"Saved topic info to {topic_info_path}")
    
    # Save topic representations
    representations_path = f'{output_dir}/topic_representations_{timestamp}.json'
    with open(representations_path, 'w') as f:
        json.dump(analysis_results['representations'], f, indent=2)
    print(f"Saved topic representations to {representations_path}")
    
    # Save temporal analysis if available
    if analysis_results['topics_over_time'] is not None:
        temporal_path = f'{output_dir}/topics_over_time_{timestamp}.csv'
        analysis_results['topics_over_time'].to_csv(temporal_path)
        print(f"Saved temporal analysis to {temporal_path}")

def run_advanced_topic_modeling():
    """Main function to run the advanced topic modeling pipeline."""
    print("Starting advanced topic modeling...")
    
    # Load data
    posts_data = fetch_posts()
    if not posts_data:
        print("No posts found in the database. Please run the data collection first.")
        return
    
    # Separate documents and timestamps
    docs = [post[0] for post in posts_data]
    timestamps = [post[1] for post in posts_data]
    
    print(f"Loaded {len(docs)} documents with timestamps.")
    
    # Perform topic modeling
    topic_model, topics, probs = perform_topic_modeling(docs, timestamps)
    
    # Analyze topics
    analysis_results = analyze_topics(topic_model, docs, timestamps)
    
    # Save results
    save_results(analysis_results)
    
    # Print summary
    print("\nTopic Modeling Summary:")
    print(f"Number of topics: {len(analysis_results['representations'])}")
    print("\nTop Topics by Size:")
    print(analysis_results['topic_info'].head())
    
    return topic_model, analysis_results

if __name__ == '__main__':
    run_advanced_topic_modeling() 