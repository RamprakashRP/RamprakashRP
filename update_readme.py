import os
import requests
import json
import urllib.request
from datetime import datetime

# GitHub API configuration
GITHUB_USERNAME = "RamprakashRP"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fetch recent public events from GitHub
def get_recent_activity():
    url = f"https://api.github.com/users/{GITHUB_USERNAME}/events/public"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            events = json.loads(response.read().decode())
            
        activity_summary = []
        for event in events[:15]: # Look at last 15 events
            event_type = event.get("type")
            repo_name = event.get("repo", {}).get("name")
            
            if event_type == "PushEvent":
                commits = event.get("payload", {}).get("commits", [])
                activity_summary.append(f"Pushed {len(commits)} commits to {repo_name}")
            elif event_type == "CreateEvent":
                activity_summary.append(f"Created a new repository or branch in {repo_name}")
            elif event_type == "WatchEvent":
                activity_summary.append(f"Starred the repository {repo_name}")
                
        return "\n".join(activity_summary) if activity_summary else "No recent public activity found."
    except Exception as e:
        print(f"Error fetching GitHub activity: {e}")
        return "Working on private stealth projects..."

# Generate summary using Gemini AI
def generate_ai_summary(activity_text):
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found. Skipping AI generation.")
        return "*The AI Agent is currently booting up... Please add GEMINI_API_KEY to GitHub Secrets!*"

    prompt = f"""
    You are an AI assistant analyzing the recent GitHub activity of Ramprakash Raja, an elite AI Engineer and Vector AI Scholar.
    Write a witty, professional, and very brief (max 2 sentences) summary of what Ramprakash has been up to lately based on this activity:
    
    {activity_text}
    
    Make it sound like an autonomous AI agent reporting on its creator's status. Do not use hashtags or emojis.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return "*The AI Agent encountered an error while processing telemetry data.*"

# Update the README.md file
def update_readme(summary):
    with open("README.md", "r", encoding="utf-8") as file:
        readme_content = file.read()

    start_marker = "<!-- AI_AGENT_SUMMARY_START -->"
    end_marker = "<!-- AI_AGENT_SUMMARY_END -->"

    start_idx = readme_content.find(start_marker)
    end_idx = readme_content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        new_readme = (
            readme_content[:start_idx + len(start_marker)]
            + "\n*" + summary + "*\n"
            + readme_content[end_idx:]
        )
        
        with open("README.md", "w", encoding="utf-8") as file:
            file.write(new_readme)
        print("README.md successfully updated!")
    else:
        print("Could not find AI agent markers in README.md")

if __name__ == "__main__":
    print("Fetching activity...")
    activity = get_recent_activity()
    print("Generating AI summary...")
    summary = generate_ai_summary(activity)
    print(f"New Summary: {summary}")
    update_readme(summary)
