"""
yahoo_auth_helper.py — Manual OAuth flow helper for Yahoo Fantasy API.

Since the Yahoo skill's venv setup requires system packages we can't install,
this is a minimal script that generates the OAuth authorization URL for you
to visit in a browser. After you authorize, Yahoo will display a code — paste
that code back and this script will exchange it for tokens and store them.
"""

import os
import json
import urllib.parse
import requests

CONSUMER_KEY = "dj0yJmk9M09lWjdMczhqeXR2JmQ9WVdrOVIyOWlkSFIwUjNJbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTQ1"
CONSUMER_SECRET = "13d2747c85e38363ffcd68ec6d4c8d51b3d44305"
REDIRECT_URI = "https://localhost:8000/callback"  # Must match Yahoo app settings

TOKEN_DIR = os.path.expanduser("~/.openclaw/credentials/yahoo-fantasy")
TOKEN_FILE = os.path.join(TOKEN_DIR, "oauth2.json")

def generate_auth_url():
    """Generate the Yahoo OAuth authorization URL."""
    params = {
        "client_id": CONSUMER_KEY,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "fspt-r",  # Fantasy Sports read-only
    }
    url = "https://api.login.yahoo.com/oauth2/request_auth?" + urllib.parse.urlencode(params)
    return url

def exchange_code_for_token(auth_code):
    """Exchange the authorization code for access/refresh tokens."""
    url = "https://api.login.yahoo.com/oauth2/get_token"
    data = {
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": auth_code,
    }
    auth = (CONSUMER_KEY, CONSUMER_SECRET)
    
    response = requests.post(url, data=data, auth=auth)
    response.raise_for_status()
    
    token_data = response.json()
    
    # Add consumer credentials for token refresh
    token_data["consumer_key"] = CONSUMER_KEY
    token_data["consumer_secret"] = CONSUMER_SECRET
    
    # Save to file
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    
    # Set restrictive permissions
    os.chmod(TOKEN_FILE, 0o600)
    
    return token_data

def load_token():
    """Load existing token if available."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None

def refresh_token():
    """Refresh expired access token using refresh token."""
    token_data = load_token()
    if not token_data or "refresh_token" not in token_data:
        return None
    
    url = "https://api.login.yahoo.com/oauth2/get_token"
    data = {
        "grant_type": "refresh_token",
        "redirect_uri": REDIRECT_URI,
        "refresh_token": token_data["refresh_token"],
    }
    auth = (CONSUMER_KEY, CONSUMER_SECRET)
    
    response = requests.post(url, data=data, auth=auth)
    response.raise_for_status()
    
    new_token = response.json()
    new_token["consumer_key"] = CONSUMER_KEY
    new_token["consumer_secret"] = CONSUMER_SECRET
    
    with open(TOKEN_FILE, "w") as f:
        json.dump(new_token, f, indent=2)
    
    os.chmod(TOKEN_FILE, 0o600)
    
    return new_token

if __name__ == "__main__":
    # Check if we already have a valid token
    existing = load_token()
    if existing and "access_token" in existing:
        print("Token file exists. Testing validity...")
        # Quick test with Yahoo API
        headers = {"Authorization": f"Bearer {existing['access_token']}"}
        test = requests.get("https://api.login.yahoo.com/openid/v1/userinfo", headers=headers)
        if test.status_code == 200:
            print("✅ Token is valid!")
            print(f"User: {test.json().get('name', 'Unknown')}")
            exit(0)
        else:
            print("Token expired, refreshing...")
            new_token = refresh_token()
            if new_token:
                print("✅ Token refreshed successfully!")
                exit(0)
    
    # No valid token — start OAuth flow
    print("=" * 60)
    print("Yahoo Fantasy API OAuth Setup")
    print("=" * 60)
    print()
    print("Step 1: Visit this URL in your browser and authorize:")
    print()
    print(generate_auth_url())
    print()
    print("Step 2: After authorizing, Yahoo will display a code.")
    print("Paste that code here and press Enter:")
    print()
    
    auth_code = input("Authorization code: ").strip()
    
    if not auth_code:
        print("No code provided. Exiting.")
        exit(1)
    
    print("Exchanging code for token...")
    token = exchange_code_for_token(auth_code)
    
    print()
    print("✅ Success! Token saved to:")
    print(f"   {TOKEN_FILE}")
    print()
    print(f"Access token expires in: {token.get('expires_in', 'unknown')} seconds")
