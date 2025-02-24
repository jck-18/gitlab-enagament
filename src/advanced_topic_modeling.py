import sqlite3
import yaml
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import pandas as pd
import time

def load_config(path='./config/config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def fetch_posts():
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

def create_advanced_topic_model(docs, timestamps=None):
    # Initialize BERT embedding model
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
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=True
    )
    
    # Fit the model
    if timestamps:
        topics, probs = topic_model.fit_transform(docs, timestamps)
    else:
        topics, probs = topic_model.fit_transform(docs)
    
    return topic_model, topics, probs

def analyze_topics(topic_model, docs, timestamps=None):
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Get topic representations
    topic_representations = {}
    for topic in topic_info['Topic']:
        if topic != -1:  # Skip outlier topic
            words, scores = zip(*topic_model.get_topic(topic))
            topic_representations[topic] = {
                'words': words[:10],
                'scores': scores[:10]
            }
    
    # Temporal analysis if timestamps provided
    if timestamps:
        topics_over_time = topic_model.topics_over_time(
            docs,
            timestamps,
            global_tuning=True
        )
    
    # Generate visualizations
    topic_model.visualize_topics()
    topic_model.visualize_hierarchy()
    
    if timestamps:
        topic_model.visualize_topics_over_time(topics_over_time)
    
    return {
        'topic_info': topic_info,
        'representations': topic_representations,
        'topics_over_time': topics_over_time if timestamps else None
    }

def save_results(analysis_results, output_dir='results'):
    import os
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save topic info
    topic_info_df = analysis_results['topic_info']
    topic_info_df.to_csv(f'{output_dir}/topic_info_{timestamp}.csv')
    
    # Save topic representations
    with open(f'{output_dir}/topic_representations_{timestamp}.json', 'w') as f:
        json.dump(analysis_results['representations'], f, indent=2)
    
    # Save temporal analysis if available
    if analysis_results['topics_over_time'] is not None:
        analysis_results['topics_over_time'].to_csv(
            f'{output_dir}/topics_over_time_{timestamp}.csv'
        )

def run_advanced_topic_modeling():
    # Load data
    posts_data = fetch_posts()
    docs = [post[0] for post in posts_data]
    timestamps = [post[1] for post in posts_data]
    
    # Create and fit model
    topic_model, topics, probs = create_advanced_topic_model(docs, timestamps)
    
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