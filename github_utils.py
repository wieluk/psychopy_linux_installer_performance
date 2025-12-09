"""Shared utilities for GitHub API interactions."""
import subprocess
import json
import time

from config import REPO


def run_gh_command(args, max_retries=3):
    """Run a GitHub CLI command with retry logic for rate limiting."""
    cmd = ["gh"] + args
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            stderr_lower = e.stderr.lower()
            
            # Check for rate limiting
            if 'rate limit' in stderr_lower or 'api rate limit exceeded' in stderr_lower:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 60  # Exponential backoff: 60s, 120s, 240s
                    print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"⚠️  Rate limit exceeded after {max_retries} retries")
                    return None
            
            # Non-rate-limit error
            print(f"⚠️  Error running gh command: {e.stderr}")
            return None
    
    return None


def categorize_asset(asset_name):
    """Categorize an asset by its filename.
    
    Returns:
        str: 'installer', 'wx_wheel', 'python_wheel', or 'other'
    """
    name_lower = asset_name.lower()
    
    # Installer files (typically .sh scripts)
    if name_lower.endswith('.sh'):
        return 'installer'
    
    # Wheel files
    if name_lower.endswith('.whl'):
        # wx wheels
        if 'wxpython' in name_lower or 'wx' in name_lower:
            return 'wx_wheel'
        # Python/PsychoPy wheels (to be ignored in early releases)
        elif 'psychopy' in name_lower or 'python' in name_lower:
            return 'python_wheel'
    
    return 'other'


def fetch_releases_with_assets():
    """Fetch GitHub releases with their assets and download counts."""
    print("ℹ️  Fetching GitHub releases with download counts...")
    
    args = [
        "api", "--paginate",
        f"repos/{REPO}/releases"
    ]
    
    output = run_gh_command(args)
    if not output:
        return []
    
    try:
        releases = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"⚠️  Could not parse releases JSON: {e}")
        return []
    
    print(f"ℹ️  Found {len(releases)} releases")
    return releases
