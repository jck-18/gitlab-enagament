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
                    
                    # First, send an acknowledgment message to show we're processing
                    response_url = payload.get('response_url')
                    if response_url:
                        requests.post(response_url, json={
                            "text": f"Generating draft response for: {title}...",
                            "response_type": "ephemeral"
                        })
                    
                    # Generate response using RAG with a more detailed prompt
                    query = f"""Create a comprehensive response to this GitLab question:
Title: {title}
Content: {content}

Please provide:
1. A friendly and professional greeting
2. A clear and concise answer to the question
3. Step-by-step instructions if applicable
4. Links to relevant GitLab documentation
5. A polite closing
"""
                    
                    # Get RAG response
                    response_data = rag.generate_response(query)
                    response_text = response_data.get('response', 'No response generated')
                    
                    # Save draft response to database
                    cursor.execute("UPDATE posts SET draft_response = ? WHERE id = ?", (response_text, post_id))
                    conn.commit()
                    
                    # Extract sources used in the response
                    sources = []
                    for doc in response_data.get('retrieved_documents', []):
                        if 'metadata' in doc and 'source' in doc['metadata']:
                            sources.append(doc['metadata']['source'])
                    
                    # Format sources as a string with links if available
                    sources_text = ""
                    if sources:
                        for source in sources:
                            # Check if source contains a URL
                            if source.startswith(('http://', 'https://', 'www.')):
                                sources_text += f"• <{source}|{source.split('/')[-1]}>\n"
                            else:
                                sources_text += f"• {source}\n"
                    else:
                        sources_text = "No specific sources used"
                    
                    # Send response back to Slack
                    return jsonify({
                        "response_type": "in_channel",
                        "text": f"Draft response for: {title}",
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": "Draft Response",
                                    "emoji": True
                                }
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*For:* <{url}|{title}>"
                                }
                            },
                            {
                                "type": "divider"
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": response_text
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
                                "type": "actions",
                                "elements": [
                                    {
                                        "type": "button",
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Run RAG Analysis",
                                            "emoji": True
                                        },
                                        "value": f"{post_id}",
                                        "action_id": "run_rag_analysis"
                                    },
                                    {
                                        "type": "button",
                                        "text": {
                                            "type": "plain_text",
                                            "text": "View on Reddit",
                                            "emoji": True
                                        },
                                        "url": url,
                                        "action_id": "view_reddit"
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
                    
                    # Generate RAG analysis with a more detailed prompt
                    query = f"""Analyze this GitLab-related post and provide insights:
Title: {title}
Content: {content}

Please provide:
1. A summary of the main issue or question
2. Key technical concepts mentioned
3. Potential solutions based on GitLab documentation
4. Any additional context that might be helpful
"""
                    
                    # Get RAG response
                    response_data = rag.generate_response(query)
                    analysis_text = response_data.get('response', 'No analysis generated')
                    
                    # Extract sources used in the analysis
                    sources = []
                    for doc in response_data.get('retrieved_documents', []):
                        if 'metadata' in doc and 'source' in doc['metadata']:
                            sources.append(doc['metadata']['source'])
                    
                    # Format sources as a string with links if available
                    sources_text = ""
                    if sources:
                        for source in sources:
                            # Check if source contains a URL
                            if source.startswith(('http://', 'https://', 'www.')):
                                sources_text += f"• <{source}|{source.split('/')[-1]}>\n"
                            else:
                                sources_text += f"• {source}\n"
                    else:
                        sources_text = "No specific sources used"
                    
                    # Send the enhanced analysis back to Slack
                    return jsonify({
                        "response_type": "in_channel",
                        "text": f"RAG Analysis for: {title}",
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": "RAG Analysis Results",
                                    "emoji": True
                                }
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Post:* <{url}|{title}>"
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
                                "type": "actions",
                                "elements": [
                                    {
                                        "type": "button",
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Generate Draft Response",
                                            "emoji": True
                                        },
                                        "value": f"{post_id}",
                                        "action_id": "generate_draft",
                                        "style": "primary"
                                    },
                                    {
                                        "type": "button",
                                        "text": {
                                            "type": "plain_text",
                                            "text": "View on Reddit",
                                            "emoji": True
                                        },
                                        "url": url,
                                        "action_id": "view_reddit"
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

@app.route('/slack/rag-query', methods=['POST'])
def rag_query():
    """
    Handle direct RAG queries from Slack slash commands.
    Example: /rag-query How do I set up GitLab CI/CD pipelines?
    """
    # Verify the request is from Slack
    request_body = request.get_data().decode('utf-8')
    timestamp = request.headers.get('X-Slack-Request-Timestamp', '')
    signature = request.headers.get('X-Slack-Signature', '')
    
    if not verify_slack_request(request_body, timestamp, signature):
        return jsonify({"error": "Invalid request signature"}), 403
    
    # Get the query from the request
    query = request.form.get('text', '')
    user_id = request.form.get('user_id', '')
    channel_id = request.form.get('channel_id', '')
    response_url = request.form.get('response_url', '')
    
    if not query:
        return jsonify({
            "response_type": "ephemeral",
            "text": "Please provide a query. Example: `/rag-query How do I set up GitLab CI/CD pipelines?`"
        })
    
    # Start processing in a separate thread
    import threading
    thread = threading.Thread(target=process_rag_query, args=(query, response_url))
    thread.daemon = True
    thread.start()
    
    # Send an immediate response to acknowledge receipt
    return jsonify({
        "response_type": "ephemeral",
        "text": f"Processing your query: *{query}*\nThis may take a few seconds..."
    })

def process_rag_query(query, response_url):
    """Process a RAG query and send the result to the response_url."""
    try:
        # Generate RAG response
        response_data = rag.generate_response(query)
        response_text = response_data.get('response', 'No response generated')
        
        # Extract sources
        sources = []
        for doc in response_data.get('retrieved_documents', []):
            if 'metadata' in doc and 'source' in doc['metadata']:
                sources.append(doc['metadata']['source'])
        
        # Format sources as a string with links if available
        sources_text = ""
        if sources:
            for source in sources:
                # Check if source contains a URL
                if source.startswith(('http://', 'https://', 'www.')):
                    sources_text += f"• <{source}|{source.split('/')[-1]}>\n"
                else:
                    sources_text += f"• {source}\n"
        else:
            sources_text = "No specific sources used"
        
        # Send the response back to Slack
        if response_url:
            payload = {
                "response_type": "in_channel",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "GitLab RAG Query Results",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Query:* {query}"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": response_text
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
                    }
                ]
            }
            requests.post(response_url, json=payload)
    except Exception as e:
        print(f"Error processing RAG query: {e}")
        # Send error message back to Slack
        if response_url:
            payload = {
                "response_type": "ephemeral",
                "text": f"Error processing your query: {str(e)}"
            }
            requests.post(response_url, json=payload)

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False) 