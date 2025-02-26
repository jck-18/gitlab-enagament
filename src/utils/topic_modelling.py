import sqlite3
import yaml
from gensim import corpora, models
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import os

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_config(path='./config/config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def fetch_posts():
    connection = sqlite3.connect('reddit_posts.db')
    cursor = connection.cursor()
    cursor.execute("SELECT content FROM posts WHERE content IS NOT NULL")
    posts = cursor.fetchall()
    connection.close()
    # Flatten list of tuples
    return [post[0] for post in posts if post[0].strip() != ""]

def preprocess_documents(documents):
    # Tokenize, remove stop words and short tokens
    return [
        [token for token in simple_preprocess(doc) if token not in stop_words and len(token) > 3]
        for doc in documents
    ]

def run_topic_modeling():
    config = load_config()
    topic_cfg = config.get('topic_modeling', {})
    num_topics = topic_cfg.get('num_topics', 5)
    passes = topic_cfg.get('passes', 10)

    documents = fetch_posts()
    if not documents:
        print("No documents found in the database for topic modeling.")
        return

    processed_docs = preprocess_documents(documents)
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=topic_cfg.get('random_state', 42))
    topics = lda_model.print_topics(num_words=5)

    print("Ranked Topics:")
    for topic in topics:
        print(topic)
    
    # Optionally, save topics to a file for later use
    save_topics(lda_model.show_topics(num_words=5, formatted=False))

def save_topics(topics_dict):
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'topics.txt')
    with open(output_path, "w") as f:
        for topic_id, words in topics_dict.items():
            f.write(f"Topic {topic_id}: {', '.join(words)}\n")

if __name__ == '__main__':
    run_topic_modeling()
