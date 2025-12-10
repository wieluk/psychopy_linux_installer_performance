#!/usr/bin/env python3
"""Fetch and display download statistics for releases."""
import json
from collections import defaultdict
from pathlib import Path

from config import REPO, DATA_DIR
from github_utils import fetch_releases_with_assets, categorize_asset


# File where release data is stored
RELEASES_FILE = DATA_DIR / "releases.json"


def display_installer_downloads(releases):
    """Display download counts for installer files in each release."""
    total_installer_downloads = 0
    releases_with_installers = 0
    
    for release in releases:
        assets = release.get('assets', [])
        
        installer_assets = [
            asset for asset in assets 
            if categorize_asset(asset['name']) == 'installer'
        ]
        
        if installer_assets:
            releases_with_installers += 1
            for asset in installer_assets:
                downloads = asset.get('download_count', 0)
                total_installer_downloads += downloads
    
    print(f"\nInstaller downloads: {total_installer_downloads:,} ({releases_with_installers} releases)")


def display_wx_wheel_downloads(releases, ignore_python_wheels=True):
    """Display total download counts for wx wheels across all releases.
    
    Args:
        releases: List of release data from GitHub API
        ignore_python_wheels: If True, ignores python wheels in the count
    """
    # Aggregate downloads by wheel filename
    wheel_downloads = defaultdict(int)
    wheel_releases = defaultdict(list)
    
    for release in releases:
        tag = release.get('tag_name', 'unknown')
        assets = release.get('assets', [])
        
        for asset in assets:
            asset_name = asset.get('name', '')
            asset_type = categorize_asset(asset_name)
            
            # Skip python wheels if requested (for early releases)
            if ignore_python_wheels and asset_type == 'python_wheel':
                continue
            
            # Only count wx wheels
            if asset_type == 'wx_wheel':
                downloads = asset.get('download_count', 0)
                wheel_downloads[asset_name] += downloads
                wheel_releases[asset_name].append({
                    'tag': tag,
                    'downloads': downloads
                })
    
    if not wheel_downloads:
        print("No wx wheel files found")
        return
    
    total_wx_downloads = sum(wheel_downloads.values())
    print(f"WX wheel downloads: {total_wx_downloads:,} ({len(wheel_downloads)} unique wheels)")


def display_summary_statistics(releases):
    """Display summary statistics for all downloads."""
    total_downloads = 0
    downloads_by_type = defaultdict(int)
    
    for release in releases:
        assets = release.get('assets', [])
        
        for asset in assets:
            asset_name = asset.get('name', '')
            downloads = asset.get('download_count', 0)
            asset_type = categorize_asset(asset_name)
            
            total_downloads += downloads
            downloads_by_type[asset_type] += downloads
    
    print(f"\nTotal: {total_downloads:,} downloads across {len(releases)} releases")
    print(f"Breakdown: Installer {downloads_by_type['installer']:,} | WX wheels {downloads_by_type['wx_wheel']:,} | Python wheels {downloads_by_type['python_wheel']:,} | Other {downloads_by_type['other']:,}")


def load_releases():
    """Load releases from saved file or fetch from GitHub if not available."""
    if RELEASES_FILE.exists():
        try:
            with open(RELEASES_FILE, 'r', encoding='utf-8') as f:
                releases = json.load(f)
            print(f"Loaded {len(releases)} releases from cache")
            return releases
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading cache, fetching from GitHub...")
    
    return fetch_releases_with_assets()


def main():
    """Main function to fetch and display download statistics."""
    print(f"Download statistics for {REPO}:")
    
    releases = load_releases()
    
    if not releases:
        print("No releases found")
        return
    
    # Display statistics
    display_installer_downloads(releases)
    display_wx_wheel_downloads(releases, ignore_python_wheels=True)
    display_summary_statistics(releases)


if __name__ == "__main__":
    main()
