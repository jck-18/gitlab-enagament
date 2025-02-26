#!/usr/bin/env python
"""
Example script demonstrating how to trigger a Slack workflow from your application.
This can be used to send data to a Slack workflow that starts with a webhook.
"""

import os
import sys
import yaml
import requests
import json
import argparse
from dotenv import load_dotenv
from src.core.together_rag import TogetherRAG

# Load environment variables
load_dotenv()

def load_config(path='./config/config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found at {path}. Please ensure the file exists.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

def trigger_slack_workflow(url, title, additional_data=None):
    """
    Trigger a Slack workflow by sending data to its webhook URL.
    
    Args:
        url: The URL to send to the workflow
        title: The title to send to the workflow
        additional_data: Optional dictionary of additional data to include
    
    Returns:
        True if successful, False otherwise
    """
    # Get workflow webhook URL from environment variable
    workflow_url = os.getenv('SLACK_WORKFLOW_WEBHOOK_URL')
    
    # If not in environment, try to load from config as fallback
    if not workflow_url:
        # Load configuration
        config = load_config()
        if not config:
            print("Failed to load configuration.")
            return False
        
        # Get workflow webhook URL from config
        workflow_url = config.get('slack', {}).get('workflow_webhook_url')
    
    if not workflow_url:
        print("Slack workflow webhook URL not configured in environment or config.yaml.")
        print("Please set the SLACK_WORKFLOW_WEBHOOK_URL environment variable.")
        return False
    
    # Build payload with the expected variables
    payload = {
        "url": url,
        "title": title
    }
    
    # Add additional data if provided
    if additional_data and isinstance(additional_data, dict):
        payload.update(additional_data)
    
    # Send request to Slack workflow webhook
    try:
        print(f"Sending data to Slack workflow: {json.dumps(payload, indent=2)}")
        response = requests.post(workflow_url, json=payload)
        
        if response.status_code == 200:
            print("Successfully triggered Slack workflow!")
            return True
        else:
            print(f"Failed to trigger Slack workflow. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"Error triggering Slack workflow: {e}")
        return False

def trigger_rag_analysis(url, title, content=None):
    """
    Trigger a RAG analysis and return the results.
    
    Args:
        url: The URL of the content to analyze
        title: The title of the content
        content: The content to analyze (optional)
    
    Returns:
        Dictionary with analysis results
    """
    # Initialize RAG system
    rag = TogetherRAG()
    
    # Prepare query
    if content:
        query = f"Analyze this GitLab-related post and provide insights: {title}\n\nContent: {content}"
    else:
        query = f"Analyze this GitLab-related post and provide insights: {title}"
    
    # Generate RAG analysis
    try:
        print(f"Generating RAG analysis for: {title}")
        response_data = rag.generate_response(query)
        
        # Extract sources used in the analysis
        sources = []
        for doc in response_data.get('retrieved_documents', []):
            if 'metadata' in doc and 'source' in doc['metadata']:
                sources.append(doc['metadata']['source'])
        
        # Return results
        return {
            "success": True,
            "title": title,
            "url": url,
            "analysis": response_data.get('response', 'No analysis generated'),
            "sources": sources
        }
    
    except Exception as e:
        print(f"Error generating RAG analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function to parse arguments and trigger the workflow."""
    parser = argparse.ArgumentParser(description='Trigger a Slack workflow with data or run RAG analysis.')
    parser.add_argument('--url', required=True, help='The URL to send to the workflow or analyze')
    parser.add_argument('--title', required=True, help='The title to send to the workflow or analyze')
    parser.add_argument('--data', help='JSON string of additional data to include')
    parser.add_argument('--content', help='Content to analyze with RAG')
    parser.add_argument('--rag', action='store_true', help='Run RAG analysis instead of triggering workflow')
    
    args = parser.parse_args()
    
    # Parse additional data if provided
    additional_data = None
    if args.data:
        try:
            additional_data = json.loads(args.data)
        except json.JSONDecodeError:
            print("Error: --data must be a valid JSON string")
            sys.exit(1)
    
    # Run RAG analysis if requested
    if args.rag:
        results = trigger_rag_analysis(
            url=args.url,
            title=args.title,
            content=args.content
        )
        
        # Print results
        print(json.dumps(results, indent=2))
        sys.exit(0 if results.get('success', False) else 1)
    
    # Otherwise trigger the workflow
    else:
        success = trigger_slack_workflow(
            url=args.url,
            title=args.title,
            additional_data=additional_data
        )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 