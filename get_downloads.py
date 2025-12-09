#!/usr/bin/env python3
"""Fetch and display download statistics for releases."""
import subprocess
import json
import time
from collections import defaultdict

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


def display_installer_downloads(releases):
    """Display download counts for installer files in each release."""
    print("\n" + "=" * 80)
    print("INSTALLER FILE DOWNLOADS PER RELEASE")
    print("=" * 80)
    
    total_installer_downloads = 0
    releases_with_installers = 0
    
    for release in releases:
        tag = release.get('tag_name', 'unknown')
        name = release.get('name', tag)
        published = release.get('published_at', 'unknown')
        assets = release.get('assets', [])
        
        installer_assets = [
            asset for asset in assets 
            if categorize_asset(asset['name']) == 'installer'
        ]
        
        if installer_assets:
            releases_with_installers += 1
            print(f"\n📦 Release: {name} ({tag})")
            print(f"   Published: {published}")
            
            for asset in installer_assets:
                downloads = asset.get('download_count', 0)
                asset_name = asset.get('name', 'unknown')
                total_installer_downloads += downloads
                print(f"   └─ {asset_name}: {downloads:,} downloads")
    
    print(f"\n{'─' * 80}")
    print(f"Total: {releases_with_installers} releases with installers")
    print(f"Total installer downloads: {total_installer_downloads:,}")
    print("=" * 80)


def display_wx_wheel_downloads(releases, ignore_python_wheels=True):
    """Display total download counts for wx wheels across all releases.
    
    Args:
        releases: List of release data from GitHub API
        ignore_python_wheels: If True, ignores python wheels in the count
    """
    print("\n" + "=" * 80)
    print("WX WHEEL DOWNLOADS (Total Across All Releases)")
    print("=" * 80)
    
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
        print("\n⚠️  No wx wheel files found in releases")
        print("=" * 80)
        return
    
    # Sort by total downloads (descending)
    sorted_wheels = sorted(
        wheel_downloads.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    total_wx_downloads = 0
    
    print(f"\nFound {len(sorted_wheels)} unique wx wheel files:")
    print()
    
    for wheel_name, total_downloads in sorted_wheels:
        total_wx_downloads += total_downloads
        num_releases = len(wheel_releases[wheel_name])
        
        print(f"🔧 {wheel_name}")
        print(f"   Total downloads: {total_downloads:,}")
        print(f"   Appears in {num_releases} release(s)")
        
        # Show breakdown by release if there are multiple
        if num_releases > 1:
            print(f"   Breakdown by release:")
            for release_info in sorted(
                wheel_releases[wheel_name], 
                key=lambda x: x['downloads'], 
                reverse=True
            ):
                print(f"      └─ {release_info['tag']}: {release_info['downloads']:,}")
        print()
    
    print(f"{'─' * 80}")
    print(f"Total wx wheel downloads: {total_wx_downloads:,}")
    print("=" * 80)


def display_summary_statistics(releases):
    """Display summary statistics for all downloads."""
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY STATISTICS")
    print("=" * 80)
    
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
    
    print(f"\nTotal releases: {len(releases)}")
    print(f"Total downloads (all assets): {total_downloads:,}")
    print(f"\nBreakdown by asset type:")
    
    for asset_type in ['installer', 'wx_wheel', 'python_wheel', 'other']:
        count = downloads_by_type.get(asset_type, 0)
        percentage = (count / total_downloads * 100) if total_downloads > 0 else 0
        print(f"  {asset_type.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    print("=" * 80)


def main():
    """Main function to fetch and display download statistics."""
    print("📊 Fetching download statistics for repository: " + REPO)
    print()
    
    releases = fetch_releases_with_assets()
    
    if not releases:
        print("❌ No releases found or error fetching releases")
        return
    
    # Display statistics
    display_installer_downloads(releases)
    display_wx_wheel_downloads(releases, ignore_python_wheels=True)
    display_summary_statistics(releases)
    
    print("\n✅ Download statistics retrieval complete!")


if __name__ == "__main__":
    main()
