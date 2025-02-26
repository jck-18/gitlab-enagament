# Slack Integration for GitLab RAG Assistant

This document provides instructions on how to set up the Slack integration for the GitLab RAG Assistant. The integration allows you to:

1. Receive notifications in Slack when new GitLab-related Reddit posts are detected
2. Generate draft responses to these posts directly from Slack
3. Create custom workflows in Slack that trigger actions in the RAG system

## Prerequisites

- A Slack workspace where you have permissions to create apps and workflows
- A server with a public IP address or domain name to host the webhook endpoint
- Python 3.8+ with the dependencies listed in `requirements.txt`

## Setup Instructions

### 1. Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps) and click "Create New App"
2. Choose "From scratch" and provide a name (e.g., "GitLab RAG Assistant") and select your workspace
3. Click "Create App"

### 2. Configure Slack App Permissions

1. In the left sidebar, click on "OAuth & Permissions"
2. Under "Bot Token Scopes", add the following scopes:
   - `chat:write` (Send messages as the app)
   - `incoming-webhook` (Post messages to specific channels)
   - `commands` (Add slash commands)
   - `reactions:write` (Add reactions to messages)

3. Click "Install to Workspace" at the top of the page and authorize the app

### 3. Create an Incoming Webhook

1. In the left sidebar, click on "Incoming Webhooks"
2. Toggle "Activate Incoming Webhooks" to On
3. Click "Add New Webhook to Workspace"
4. Select the channel where you want to receive notifications and click "Allow"
5. Copy the Webhook URL that is generated

### 4. Configure Your Application

1. Open `config/config.yaml` and update the `slack` section:
   ```yaml
   slack:
     webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"  # Paste your webhook URL here
   ```

2. Create a `.env` file in the root directory and add your Slack signing secret:
   ```
   SLACK_SIGNING_SECRET=your_slack_signing_secret
   ```
   You can find your signing secret in the "Basic Information" section of your Slack app settings.

### 5. Set Up the Webhook Server

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask server:
   ```bash
   python src/slack_webhook_server.py
   ```
   
   For production, it's recommended to use Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 src.slack_webhook_server:app
   ```

3. Make your server publicly accessible (using a reverse proxy like Nginx, or a service like ngrok for testing)

### 6. Configure Slack Interactivity

1. In your Slack app settings, go to "Interactivity & Shortcuts"
2. Toggle "Interactivity" to On
3. Set the Request URL to your public server URL + `/slack/events` (e.g., `https://your-server.com/slack/events`)
4. Click "Save Changes"

## Creating a Slack Workflow with Webhooks

To create a custom workflow in Slack that triggers actions in your RAG system:

1. In Slack, click on your workspace name in the top left
2. Select "Tools" > "Workflow Builder"
3. Click "Create" to create a new workflow
4. Select "Start from scratch"
5. Give your workflow a name (e.g., "GitLab Question Analyzer")
6. For the trigger, select "Webhook"
7. Copy the webhook URL provided by Slack
8. Configure your variables:
   - Add a variable named `url` (type: text)
   - Add a variable named `title` (type: text)
9. Add steps to your workflow (e.g., post a message to a channel)
10. Publish your workflow

## Integrating the Slack Workflow with Your Application

To send data to your Slack workflow:

1. Add the Slack workflow webhook URL to your configuration:
   ```yaml
   slack:
     webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
     workflow_webhook_url: "https://hooks.slack.com/triggers/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"
   ```

2. Use the `requests` library to send data to the workflow webhook:
   ```python
   import requests
   import json
   
   def trigger_slack_workflow(url, title):
       workflow_url = config.get('slack', {}).get('workflow_webhook_url')
       if not workflow_url:
           return
           
       payload = {
           "url": url,
           "title": title
       }
       
       response = requests.post(workflow_url, json=payload)
       if response.status_code != 200:
           print(f"Failed to trigger Slack workflow: {response.text}")
   ```

## Testing the Integration

1. Run the Reddit retriever to fetch posts and send them to Slack:
   ```bash
   python src/reddit_retriever.py
   ```

2. Check your Slack channel for new notifications
3. Click the "Generate Draft Response" button to test the interactive functionality

4. Manually trigger a workflow:
   ```bash
   python src/trigger_slack_workflow.py --url "https://gitlab.com/docs/ci-cd" --title "How to set up GitLab CI/CD"
   ```

## Troubleshooting

- **Webhook Not Working**: Check that your server is publicly accessible and that the URL in Slack's interactivity settings is correct
- **Authentication Errors**: Verify that your signing secret in the `.env` file matches the one in your Slack app settings
- **Missing Messages**: Ensure that your app has the correct permissions and is installed to the workspace
- **Button Actions Not Working**: Check the Slack event logs in your app settings for error messages
- **Workflow Not Triggering**: Verify that your payload matches the expected variables (`url` and `title`)

## Security Considerations

- Always verify incoming requests from Slack using the signing secret
- Use HTTPS for all webhook endpoints
- Regularly rotate your Slack app credentials
- Be careful about what information you send to Slack, especially if it contains sensitive data 