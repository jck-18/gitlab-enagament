import yaml
import praw
import requests
import os
from sqlite3 import connect
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def send_to_slack(post, config):
    """
    Send a Reddit post to Slack using a webhook.
    
    Args:
        post: The Reddit post object
        config: The loaded configuration
    """
    # Get Slack webhook URL from environment variable first, then fallback to config
    slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    # If not in environment, try to get from config as fallback
    if not slack_webhook_url and config:
        slack_webhook_url = config.get('slack', {}).get('webhook_url')
    
    if not slack_webhook_url:
        print("Slack webhook URL not configured in environment or config. Skipping Slack notification.")
        return
    
    # Build a message payload using Slack's Block Kit for rich formatting
    payload = {
        "text": f"New GitLab Reddit post: *{post.title}*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{post.title}*\n<{post.url}|View on Reddit>\nPosted: {datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Content: {post.selftext[:500]}..." if len(post.selftext) > 500 else f"Content: {post.selftext}" if post.selftext else "No content"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Generate Draft Response"
                        },
                        "value": f"{post.id}",
                        "action_id": "generate_draft"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "RAG Analysis"
                        },
                        "value": f"{post.id}",
                        "action_id": "run_rag_analysis"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(slack_webhook_url, json=payload)
        if response.status_code == 200:
            print(f"Successfully sent post '{post.title}' to Slack")
        else:
            print(f"Failed to send Slack notification: {response.text}")
    except Exception as e:
        print(f"Error sending Slack notification: {e}")
    
    # Also trigger the Slack workflow if configured
    trigger_slack_workflow(post, config)

def trigger_slack_workflow(post, config):
    """
    Trigger a Slack workflow with the post data.
    
    Args:
        post: The Reddit post object
        config: The loaded configuration
    """
    # Get Slack workflow webhook URL from environment variable first, then fallback to config
    workflow_webhook_url = os.getenv('SLACK_WORKFLOW_WEBHOOK_URL')
    
    # If not in environment, try to get from config as fallback
    if not workflow_webhook_url and config:
        workflow_webhook_url = config.get('slack', {}).get('workflow_webhook_url')
    
    if not workflow_webhook_url:
        print("Slack workflow webhook URL not configured in environment or config. Skipping workflow trigger.")
        return
    
    # Build a simple payload with the variables expected by the Slack workflow
    payload = {
        "url": post.url,
        "title": post.title
    }
    
    try:
        response = requests.post(workflow_webhook_url, json=payload)
        if response.status_code == 200:
            print(f"Successfully triggered Slack workflow for post '{post.title}'")
        else:
            print(f"Failed to trigger Slack workflow: {response.text}")
    except Exception as e:
        print(f"Error triggering Slack workflow: {e}")

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
    day_ago = current_time - timedelta(days=int(reddit_cfg.get('days_back', 1)))  # default to 1 day if not specified

    for subreddit in subreddits:
        sub = reddit.subreddit(subreddit)
        print(f"Fetching recent posts from r/{subreddit}...")
        for post in sub.new(limit=5):
            # Skip posts older than specified days
            post_time = datetime.fromtimestamp(post.created_utc)
            if post_time < day_ago:
                continue
                
            # Check if any keyword is in the title
            if any(kw.lower() in post.title.lower() for kw in keywords):
                content = post.selftext if hasattr(post, 'selftext') else ""
                save_post_to_db(post, content)
                print(f"Saved post: {post.title}")
                
                # Trigger Slack webhook with relevant data
                send_to_slack(post, config)

if __name__ == '__main__':
    retrieve_reddit_posts()
