#!/usr/bin/env python3
"""Fetch and save release download statistics."""
import sys
import json
from pathlib import Path

from config import REPO, DATA_DIR
from github_utils import fetch_releases_with_assets


# File to store release data
RELEASES_FILE = DATA_DIR / "releases.json"


def save_releases_data(releases):
    """Save release data to JSON file."""
    if not releases:
        print("⚠️  No releases to save")
        return False
    
    # Save the complete release data
    with open(RELEASES_FILE, 'w', encoding='utf-8') as f:
        json.dump(releases, f, indent=2)
    
    print(f"✅ Saved {len(releases)} releases to {RELEASES_FILE}")
    return True


def main():
    """Main function to fetch and save release data."""
    print("🚀 Starting release data fetch...")
    print(f"📊 Repository: {REPO}")
    print()
    
    releases = fetch_releases_with_assets()
    
    if not releases:
        print("❌ No releases found or error fetching releases")
        sys.exit(1)
    
    # Save the release data
    if save_releases_data(releases):
        print(f"✅ Release data fetch complete! Total releases: {len(releases)}")
        sys.exit(0)
    else:
        print("❌ Failed to save release data")
        sys.exit(1)


if __name__ == "__main__":
    main()
