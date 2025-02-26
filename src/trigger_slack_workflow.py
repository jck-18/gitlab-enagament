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
    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration.")
        return False
    
    # Get workflow webhook URL
    workflow_url = config.get('slack', {}).get('workflow_webhook_url')
    if not workflow_url:
        print("Slack workflow webhook URL not configured in config.yaml.")
        print("Please add 'workflow_webhook_url' under the 'slack' section.")
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

def main():
    """Main function to parse arguments and trigger the workflow."""
    parser = argparse.ArgumentParser(description='Trigger a Slack workflow with data.')
    parser.add_argument('--url', required=True, help='The URL to send to the workflow')
    parser.add_argument('--title', required=True, help='The title to send to the workflow')
    parser.add_argument('--data', help='JSON string of additional data to include')
    
    args = parser.parse_args()
    
    # Parse additional data if provided
    additional_data = None
    if args.data:
        try:
            additional_data = json.loads(args.data)
        except json.JSONDecodeError:
            print("Error: --data must be a valid JSON string")
            sys.exit(1)
    
    # Trigger the workflow
    success = trigger_slack_workflow(
        url=args.url,
        title=args.title,
        additional_data=additional_data
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 