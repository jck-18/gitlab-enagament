import yaml
import praw
from sqlite3 import connect
from datetime import datetime, timedelta

def load_config(path='./config/config.yaml'):
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found at {path}. Please ensure the file exists.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

def save_post_to_db(post, content):
    connection = connect('reddit_posts.db')
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            url TEXT,
            draft_response TEXT
        );
    ''')
    cursor.execute('''
        INSERT OR IGNORE INTO posts (id, title, content, url, draft_response) VALUES (?, ?, ?, ?, ?)
    ''', (post.id, post.title, content, post.url, ""))
    connection.commit()
    connection.close()

def retrieve_reddit_posts():
    config = load_config()
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    reddit_cfg = config.get('reddit', {})
    company_cfg = config.get('company', {})

    reddit = praw.Reddit(
        client_id=reddit_cfg.get('client_id'),
        client_secret=reddit_cfg.get('client_secret'),
        user_agent=reddit_cfg.get('user_agent')
    )

    subreddits = company_cfg.get('subreddits', [])
    keywords = company_cfg.get('keywords', [])
    
    # Add time filter parameters
    current_time = datetime.utcnow()
    day_ago = current_time - timedelta(days=int(reddit_cfg.get('days_back')))

    for subreddit in subreddits:
        sub = reddit.subreddit(subreddit)
        print(f"Fetching recent posts from r/{subreddit}...")
        for post in sub.new(limit=100):
            # Skip posts older than 24 hours
            post_time = datetime.fromtimestamp(post.created_utc)
            if post_time < day_ago:
                continue
                
            # Check if any keyword is in the title
            if any(kw.lower() in post.title.lower() for kw in keywords):
                content = post.selftext if hasattr(post, 'selftext') else ""
                save_post_to_db(post, content)
                print(f"Saved post: {post.title}")

if __name__ == '__main__':
    retrieve_reddit_posts()
