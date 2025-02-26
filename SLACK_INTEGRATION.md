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

1. Create a `.env` file in the root directory with your Slack credentials:
   ```
   SLACK_WEBHOOK_URL=your_slack_webhook_url
   SLACK_WORKFLOW_WEBHOOK_URL=your_slack_workflow_webhook_url
   SLACK_SIGNING_SECRET=your_slack_signing_secret
   ```
   You can find your signing secret in the "Basic Information" section of your Slack app settings.

2. For convenience, you can copy the example file:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file with your actual credentials.

3. **IMPORTANT**: Never commit your `.env` file to version control. It's already added to `.gitignore`.

4. The application will automatically load these environment variables using the `dotenv` package.

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

1. Add the Slack workflow webhook URL to your environment variables:
   ```
   SLACK_WORKFLOW_WEBHOOK_URL=your_slack_workflow_webhook_url
   ```

2. The application will automatically use this environment variable to send data to the workflow:
   ```python
   import requests
   import json
   import os
   
   def trigger_slack_workflow(url, title):
       workflow_url = os.getenv('SLACK_WORKFLOW_WEBHOOK_URL')
       if not workflow_url:
           print("Slack workflow webhook URL not configured in environment variables.")
           return False
           
       payload = {
           "url": url,
           "title": title
       }
       
       response = requests.post(workflow_url, json=payload)
       if response.status_code == 200:
           print("Successfully triggered Slack workflow!")
           return True
       else:
           print(f"Failed to trigger Slack workflow: {response.text}")
           return False
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

## Using the RAG Analysis Button

The integration now includes a "RAG Analysis" button under each Reddit post notification in Slack. This button triggers the Retrieval-Augmented Generation (RAG) pipeline to analyze the post and provide insights based on your knowledge base.

### How It Works

1. When a Reddit post is sent to Slack, it includes two buttons:
   - "Generate Draft Response" - Creates a draft response to the post
   - "RAG Analysis" - Performs an in-depth analysis using the RAG system

2. Clicking the "RAG Analysis" button:
   - Triggers the RAG pipeline to analyze the post content
   - Retrieves relevant information from your knowledge base
   - Generates an analysis with insights about the post
   - Shows the sources used in the analysis

3. The analysis is posted in the Slack channel for everyone to see

### Running RAG Analysis Manually

You can also trigger a RAG analysis manually using the command line:

```bash
python src/trigger_slack_workflow.py --rag --url "https://example.com/post" --title "Post Title" --content "Post content to analyze"
```

This will run the RAG analysis and print the results to the console.

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

### Protecting Sensitive Information

1. **Environment Variables**: Store all sensitive information in environment variables instead of configuration files:
   ```
   SLACK_WEBHOOK_URL=your_webhook_url
   SLACK_WORKFLOW_WEBHOOK_URL=your_workflow_webhook_url
   SLACK_SIGNING_SECRET=your_signing_secret
   ```

2. **Never Commit Secrets**: Ensure that files containing secrets (`.env`, `config/config.yaml`) are in your `.gitignore` file.

3. **Webhook URL Security**: Slack webhook URLs contain authentication tokens. If these URLs are exposed:
   - They can be used by anyone to post messages to your Slack workspace
   - They should be treated as secrets and never committed to version control
   - If a webhook URL is accidentally exposed, regenerate it immediately in the Slack app settings

4. **Ngrok Security**: When using ngrok for development:
   - The generated URLs provide public access to your local server
   - These URLs should never be shared publicly or committed to version control
   - For production, use a proper hosting environment with HTTPS and authentication

5. **Signing Secret Verification**: Always verify incoming requests from Slack using the signing secret to prevent spoofing:
   ```python
   def verify_slack_request(request_data, timestamp, signature):
       slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET")
       base_string = f"v0:{timestamp}:{request_data}"
       my_signature = 'v0=' + hmac.new(
           slack_signing_secret.encode(),
           base_string.encode(),
           hashlib.sha256
       ).hexdigest()
       return hmac.compare_digest(my_signature, signature)
   ```

6. **Regenerate Compromised Credentials**: If you suspect any of your Slack credentials have been compromised:
   - Regenerate your webhook URLs in the Slack app settings
   - Rotate your signing secret by regenerating your app
   - Update your environment variables with the new credentials 