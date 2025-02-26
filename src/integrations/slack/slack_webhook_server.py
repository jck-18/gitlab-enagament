import os
import json
import hmac
import hashlib
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import sqlite3
from src.core.together_rag import TogetherRAG
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load configuration
def load_config():
    try:
        with open('config/config.yaml', 'r') as f:
            import yaml
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

config = load_config()

# Initialize RAG system
rag = TogetherRAG()

# Verify Slack request signature
def verify_slack_request(request_data, timestamp, signature):
    slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    if not slack_signing_secret:
        print("Warning: SLACK_SIGNING_SECRET not set. Request verification disabled.")
        return True
        
    # Create a base string by concatenating version, timestamp, and request body
    base_string = f"v0:{timestamp}:{request_data}"
    
    # Create a signature using the signing secret
    my_signature = 'v0=' + hmac.new(
        slack_signing_secret.encode(),
        base_string.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures
    return hmac.compare_digest(my_signature, signature)

@app.route('/slack/events', methods=['POST'])
def slack_events():
    # Get request data
    request_body = request.get_data().decode('utf-8')
    timestamp = request.headers.get('X-Slack-Request-Timestamp', '')
    signature = request.headers.get('X-Slack-Signature', '')
    
    # Verify request is from Slack
    if not verify_slack_request(request_body, timestamp, signature):
        return jsonify({"error": "Invalid request signature"}), 403
    
    # Check for request age (prevent replay attacks)
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return jsonify({"error": "Request too old"}), 403
    
    # Parse the request payload
    payload = json.loads(request.form.get('payload', '{}'))
    
    # Handle different types of interactions
    if payload.get('type') == 'block_actions':
        for action in payload.get('actions', []):
            # Handle Generate Draft Response button
            if action.get('action_id') == 'generate_draft':
                post_id = action.get('value')
                
                # Get post from database
                conn = sqlite3.connect('reddit_posts.db')
                cursor = conn.cursor()
                cursor.execute("SELECT title, content, url FROM posts WHERE id = ?", (post_id,))
                post_data = cursor.fetchone()
                
                if post_data:
                    title, content, url = post_data
                    
                    # Generate response using RAG
                    query = f"How to respond to this GitLab question: {title}"
                    response_data = rag.generate_response(query)
                    response_text = response_data.get('response', 'No response generated')
                    
                    # Save draft response to database
                    cursor.execute("UPDATE posts SET draft_response = ? WHERE id = ?", (response_text, post_id))
                    conn.commit()
                    
                    # Send response back to Slack
                    return jsonify({
                        "response_type": "in_channel",
                        "text": f"Draft response for: {title}",
                        "blocks": [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Draft response for:* {title}\n\n{response_text}"
                                }
                            },
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "mrkdwn",
                                        "text": f"<{url}|View original post on Reddit>"
                                    }
                                ]
                            }
                        ]
                    })
                
                conn.close()
            
            # Handle RAG Analysis button
            elif action.get('action_id') == 'run_rag_analysis':
                post_id = action.get('value')
                
                # Get post from database
                conn = sqlite3.connect('reddit_posts.db')
                cursor = conn.cursor()
                cursor.execute("SELECT title, content, url FROM posts WHERE id = ?", (post_id,))
                post_data = cursor.fetchone()
                
                if post_data:
                    title, content, url = post_data
                    
                    # First, send an acknowledgment message to show we're processing
                    response_url = payload.get('response_url')
                    if response_url:
                        requests.post(response_url, json={
                            "text": f"Processing RAG analysis for: {title}...",
                            "response_type": "ephemeral"
                        })
                    
                    # Generate RAG analysis
                    query = f"Analyze this GitLab-related post and provide insights: {title}\n\nContent: {content}"
                    response_data = rag.generate_response(query)
                    analysis_text = response_data.get('response', 'No analysis generated')
                    
                    # Extract sources used in the analysis
                    sources = []
                    for doc in response_data.get('retrieved_documents', []):
                        if 'metadata' in doc and 'source' in doc['metadata']:
                            sources.append(doc['metadata']['source'])
                    
                    # Format sources as a string
                    sources_text = "\n".join([f"• {source}" for source in sources]) if sources else "No specific sources used"
                    
                    # Send the analysis back to Slack
                    return jsonify({
                        "response_type": "in_channel",
                        "text": f"RAG Analysis for: {title}",
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": "RAG Analysis Results"
                                }
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Post:* {title}"
                                }
                            },
                            {
                                "type": "divider"
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": analysis_text
                                }
                            },
                            {
                                "type": "divider"
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": "*Sources Used:*\n" + sources_text
                                }
                            },
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "mrkdwn",
                                        "text": f"<{url}|View original post on Reddit>"
                                    }
                                ]
                            }
                        ]
                    })
                
                conn.close()
    
    # Default response
    return jsonify({"message": "Received"}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False) 