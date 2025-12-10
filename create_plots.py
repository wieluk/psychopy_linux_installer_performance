#!/usr/bin/env python3
"""Generate performance plots from the collected data."""
import subprocess
import json
import time
from itertools import cycle
from matplotlib.lines import Line2D
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from config import SCRIPT_DIR, CSV_FILE, PLOTS_DIR, REPO, PlotConfig
from github_utils import run_gh_command, categorize_asset, fetch_releases_with_assets


def fetch_releases():
    """Fetch GitHub releases for the repository."""
    print("ℹ️  Fetching GitHub releases...")
    
    args = [
        "api", "--paginate",
        "--jq", ".[] | {tag_name: .tag_name, published_at: .published_at, name: .name}",
        f"repos/{REPO}/releases"
    ]
    
    output = run_gh_command(args)
    if not output:
        return []
    
    releases = []
    for line in output.split('\n'):
        if line.strip():
            try:
                releases.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"⚠️  Could not parse release JSON: {e}")
    
    print(f"ℹ️  Found {len(releases)} releases")
    return releases


def smart_run_ticks(total_runs, run_ids, run_dates=None, max_ticks=20, mode='short'):
    """Generate smart run ticks with multiple display modes.
    
    Args:
        total_runs: Total number of runs
        run_ids: List of run IDs
        run_dates: Optional list of dates corresponding to run_ids
        max_ticks: Maximum number of ticks to display
        mode: 'short' (last 4 digits), 'full' (full ID), or 'date' (date labels)
    """
    if total_runs <= 10:
        tick_step = 1
    elif total_runs <= 50:
        tick_step = max(1, total_runs // 10)
    else:
        tick_step = max(1, total_runs // max_ticks)
    
    tick_indices = [i for i in range(0, total_runs, tick_step) if i < total_runs]
    
    if tick_indices and (total_runs - 1 - tick_indices[-1]) > tick_step // 2:
        tick_indices.append(total_runs - 1)
    
    tick_labels = []
    for i in tick_indices:
        if i < len(run_ids):
            if mode == 'date' and run_dates is not None and i < len(run_dates):
                # Show date instead of run ID
                date = run_dates[i]
                label = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            elif mode == 'full':
                # Full run ID
                label = str(run_ids[i])
            else:  # mode == 'short'
                # Last 4 digits for readability
                run_id_str = str(run_ids[i])
                label = run_id_str[-4:] if len(run_id_str) > 4 else run_id_str
            tick_labels.append(label)
    
    return tick_indices, tick_labels


def optimize_release_labels(release_markers, total_runs):
    """Optimize release label positioning with improved overlap avoidance."""
    if not release_markers:
        return []
    
    releases_with_pos = [
        {'x_pos': r['run_index'], 'tag': r['tag'], 'date': r['date']}
        for r in release_markers
    ]
    releases_with_pos.sort(key=lambda x: x['x_pos'])
    
    # Use more height levels for better spacing
    heights = np.linspace(0.95, 0.70, PlotConfig.RELEASE_HEIGHT_LEVELS)
    min_distance = max(3, total_runs // 15)  # Stricter minimum distance
    
    optimized_releases = []
    last_positions = []  # Track recent label positions to avoid conflicts
    
    for release in releases_with_pos:
        # Find the best height level with no recent conflicts
        best_height = None
        for height in heights:
            # Check if this height level is clear of conflicts
            conflicts = [p for p in last_positions 
                        if abs(p['height'] - height) < 0.03 and  # Same height level
                        abs(p['x_pos'] - release['x_pos']) < min_distance]
            
            if not conflicts:
                best_height = height
                break
        
        # If all heights are taken, use the topmost with smaller font
        if best_height is None:
            best_height = heights[0]
            fontsize = PlotConfig.RELEASE_FONT_SIZE_TINY
        else:
            # Adjust font size based on crowding
            fontsize = PlotConfig.RELEASE_FONT_SIZE_SMALL if len(releases_with_pos) > 10 else PlotConfig.RELEASE_FONT_SIZE
        
        optimized_releases.append({
            **release,
            'height': best_height,
            'fontsize': fontsize
        })
        
        # Track this position and clean up old positions
        last_positions.append({'x_pos': release['x_pos'], 'height': best_height})
        last_positions = [p for p in last_positions 
                         if release['x_pos'] - p['x_pos'] < min_distance * 2]
    
    return optimized_releases
    optimized_releases = []
    last_x = -min_distance
    height_idx = 0
    
    for release in releases_with_pos:
        if release['x_pos'] - last_x < min_distance:
            height_idx = (height_idx + 1) % len(heights)
        else:
            height_idx = 0
        
        optimized_releases.append({
            **release,
            'height': heights[height_idx]
        })
        last_x = release['x_pos']
    
    return optimized_releases


def prepare_release_markers(releases, run_dates_mapping, max_releases=None):
    """Prepare release markers for plotting by finding closest run indices to release dates.
    Only includes releases that occur on or after the first run date."""
    if not releases or not run_dates_mapping:
        return []
    
    # Use PlotConfig default if not specified
    if max_releases is None:
        max_releases = PlotConfig.MAX_RELEASES_TO_SHOW
    
    # Get the first run date to filter out earlier releases
    first_run_date = min(run_dates_mapping.values())
    if hasattr(first_run_date, 'tz') and first_run_date.tz is not None:
        first_run_date_naive = first_run_date.tz_convert('UTC').tz_localize(None)
    else:
        first_run_date_naive = first_run_date
    
    release_markers = []
    for release in releases:
        try:
            release_date = pd.to_datetime(release['published_at'])
            # Ensure release_date is timezone-naive for comparison
            if release_date.tz is not None:
                release_date = release_date.tz_convert('UTC').tz_localize(None)
            
            # Skip releases that are before the first run date
            if release_date < first_run_date_naive:
                continue
            
            # Find the run closest to this release date
            closest_run_index = None
            min_time_diff = float('inf')
            
            for run_index, run_date in run_dates_mapping.items():
                # Ensure run_date is timezone-naive for comparison
                if hasattr(run_date, 'tz') and run_date.tz is not None:
                    run_date_naive = run_date.tz_convert('UTC').tz_localize(None)
                else:
                    run_date_naive = run_date
                
                time_diff = abs((release_date - run_date_naive).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_run_index = run_index
            
            # Only include if the release is within a reasonable time range (30 days)
            if closest_run_index is not None and min_time_diff <= 30 * 24 * 3600:  # 30 days in seconds
                closest_run_date = run_dates_mapping[closest_run_index]
                if hasattr(closest_run_date, 'date'):
                    closest_run_date_formatted = closest_run_date.date()
                else:
                    closest_run_date_formatted = closest_run_date
                
                release_markers.append({
                    'run_index': closest_run_index,
                    'tag': release['tag_name'],
                    'name': release.get('name', release['tag_name']),
                    'date': release_date.date(),
                    'closest_run_date': closest_run_date_formatted
                })
        except (ValueError, KeyError) as e:
            print(f"⚠️  Could not parse release date for {release.get('tag_name', 'unknown')}: {e}")
    
    sorted_markers = sorted(release_markers, key=lambda x: x['run_index'])
    
    # Limit to maximum number of releases with smart selection
    if len(sorted_markers) > max_releases:
        # Always include oldest and newest
        selected = [sorted_markers[0], sorted_markers[-1]]
        
        # Calculate how many more we can include
        remaining_slots = max_releases - 2
        
        if remaining_slots > 0:
            # Select evenly spaced releases from the middle
            middle_releases = sorted_markers[1:-1]
            if len(middle_releases) <= remaining_slots:
                selected.extend(middle_releases)
            else:
                # Pick evenly spaced indices
                step = len(middle_releases) / (remaining_slots + 1)
                indices = [int(i * step) for i in range(1, remaining_slots + 1)]
                selected.extend([middle_releases[i] for i in indices])
        
        # Re-sort by run_index
        sorted_markers = sorted(selected, key=lambda x: x['run_index'])
    
    return sorted_markers


def sanitize_filename(name):
    """Sanitize names for use as filenames."""
    return name.replace("/", "_").replace(", ", "_").replace(" ", "_")


def load_data():
    """Load the performance data."""
    if not CSV_FILE.exists():
        print(f"❌ No data file found at {CSV_FILE}")
        return None

    # Read CSV with python_version as string to preserve "3.10"
    df = pd.read_csv(CSV_FILE, dtype={'python_version': str})
    df["started"] = pd.to_datetime(df["started_at"])
    df["completed"] = pd.to_datetime(df["completed_at"])
    df = df.sort_values("started")

    print(f"ℹ️  Loaded {len(df)} records from {CSV_FILE}")
    print(f"ℹ️  Data spans {df['started'].dt.date.nunique()} distinct dates")
    return df


def calculate_run_average(df):
    """Calculate average for each run_id."""
    if df.empty:
        return pd.DataFrame()
    
    return df.groupby('run_id').agg({
        'duration_m': 'mean',
        'started': 'first'  # Keep the first start time for each run_id
    }).reset_index()


def calculate_trend_line(x_data, y_data):
    """Calculate linear trend line using least squares regression."""
    if len(x_data) < 2:
        return [], []
    
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean, y_clean = np.array(x_data)[valid_mask], np.array(y_data)[valid_mask]
    
    if len(x_clean) < 2:
        return [], []
    
    coeffs = np.polyfit(x_clean, y_clean, 1)
    x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_trend = np.polyval(coeffs, x_trend)
    
    return x_trend, y_trend


def setup_plot_data(df_data):
    """Setup common plot data: run_ids, run mapping, and scatter positions."""
    # Get unique run_ids in chronological order
    unique_runs = df_data.groupby('run_id')['started'].first().sort_values()
    run_ids = unique_runs.index.tolist()
    run_to_idx = {run_id: i for i, run_id in enumerate(run_ids)}
    
    # Create mapping of run index to date for release markers
    run_dates_mapping = {i: date for i, date in enumerate(unique_runs.values)}
    
    # Calculate scatter positions with slight offsets for multiple variants per run
    all_run_ids = df_data['run_id'].tolist()
    run_counts = {run_id: all_run_ids.count(run_id) for run_id in run_ids}
    
    x_positions = []
    run_counters = {run_id: 0 for run_id in run_ids}
    
    for run_id in all_run_ids:
        run_idx = run_to_idx[run_id]
        count = run_counts[run_id]
        current = run_counters[run_id]
        
        if count == 1:
            offset = 0
        else:
            # Spread variants around the run position
            offsets = np.linspace(-0.3, 0.3, count)
            offset = offsets[current]
        
        x_positions.append(run_idx + offset)
        run_counters[run_id] += 1
    
    return run_ids, run_to_idx, run_dates_mapping, x_positions


def add_run_average_line(ax, df_variant, run_to_idx):
    """Add run average and trend lines to a plot."""
    if df_variant.empty:
        return
    
    run_avg = calculate_run_average(df_variant)
    if run_avg.empty:
        return
    
    ra_x, ra_y = [], []
    for _, row in run_avg.iterrows():
        run_id = row['run_id']
        if run_id in run_to_idx:
            ra_x.append(run_to_idx[run_id])
            ra_y.append(row['duration_m'])
    
    if ra_x and ra_y:
        ax.plot(ra_x, ra_y, color='blue', linewidth=2, alpha=0.8, 
               label='Run Average', zorder=4)
        
        if len(ra_x) >= 2:
            trend_x, trend_y = calculate_trend_line(ra_x, ra_y)
            if len(trend_x) > 0:
                ax.plot(trend_x, trend_y, color='green', linewidth=2, 
                       linestyle='--', alpha=0.8, label='Trend Line', zorder=4)



def add_release_markers(ax, release_markers):
    """Add release markers to a plot with optimized positioning."""
    optimized_releases = optimize_release_labels(release_markers, len(release_markers))
    for release in optimized_releases:
        ax.axvline(x=release['x_pos'], color='red', linestyle='--', alpha=0.7, zorder=2)
        ax.text(release['x_pos'], ax.get_ylim()[1] * release['height'], 
               f"{release['tag']}\n({release['date']})", 
               rotation=90, ha='right', va='top', 
               fontsize=release.get('fontsize', PlotConfig.RELEASE_FONT_SIZE), 
               color='red', alpha=0.8)


def setup_axis_formatting(ax, run_ids, title, max_ticks=20, run_dates=None, mode='short'):
    """Setup common axis formatting for plots.
    
    Args:
        ax: Matplotlib axis
        run_ids: List of run IDs
        title: Plot title
        max_ticks: Maximum number of ticks
        run_dates: Optional list of dates for date-based labels
        mode: Tick label mode ('short', 'full', or 'date')
    """
    ax.set_title(title)
    ax.set_xlabel("Run ID" if mode != 'date' else "Date")
    ax.set_ylabel("Duration (minutes)")
    
    tick_indices, tick_labels = smart_run_ticks(len(run_ids), run_ids, run_dates, max_ticks, mode)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=1)


def create_distribution_plot(df):
    """Create box plots showing duration distributions by category."""
    print("ℹ️  Creating distribution box plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=PlotConfig.DISTRIBUTION_PLOT_SIZE)
    fig.suptitle('Installation Duration Distribution by Category', fontsize=16, y=0.98)
    
    # Box plot by Python version
    python_versions = sorted(df['python_version'].unique())
    python_data = [df[df['python_version'] == pv]['duration_m'].values for pv in python_versions]
    axes[0].boxplot(python_data, tick_labels=python_versions)
    axes[0].set_title('By Python Version')
    axes[0].set_xlabel('Python Version')
    axes[0].set_ylabel('Duration (minutes)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Box plot by PsychoPy version
    psychopy_versions = sorted(df['psychopy_version'].unique())
    psychopy_data = [df[df['psychopy_version'] == ppv]['duration_m'].values for ppv in psychopy_versions]
    axes[1].boxplot(psychopy_data, tick_labels=psychopy_versions)
    axes[1].set_title('By PsychoPy Version')
    axes[1].set_xlabel('PsychoPy Version')
    axes[1].set_ylabel('Duration (minutes)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Box plot by OS
    os_names = sorted(df['os'].unique())
    os_data = [df[df['os'] == os_name]['duration_m'].values for os_name in os_names]
    axes[2].boxplot(os_data, tick_labels=os_names)
    axes[2].set_title('By Operating System')
    axes[2].set_xlabel('Operating System')
    axes[2].set_ylabel('Duration (minutes)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    dist_path = PLOTS_DIR / "distribution_boxplots.png"
    fig.savefig(dist_path, dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Distribution box plots → {dist_path}")
    return dist_path


def create_performance_heatmap(df):
    """Create heatmap showing average durations across configurations."""
    print("ℹ️  Creating performance heatmap...")
    
    # Create pivot table with OS+Python as rows and PsychoPy version as columns
    pivot = df.pivot_table(
        values='duration_m',
        index=['os', 'python_version'],
        columns='psychopy_version',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=PlotConfig.HEATMAP_PLOT_SIZE)
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels([f"{os}/{py}" for os, py in pivot.index])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Duration (minutes)')
    
    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                text_color = "white" if value > pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f'{value:.1f}',
                       ha="center", va="center", color=text_color, fontsize=9)
    
    ax.set_title('Average Installation Duration Heatmap\n(OS/Python × PsychoPy Version)', 
                fontsize=14, pad=20)
    ax.set_xlabel('PsychoPy Version', fontsize=12)
    ax.set_ylabel('OS / Python Version', fontsize=12)
    
    plt.tight_layout()
    
    heatmap_path = PLOTS_DIR / "performance_heatmap.png"
    fig.savefig(heatmap_path, dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Performance heatmap → {heatmap_path}")
    return heatmap_path


def create_variant_plots(df, releases):
    """Create plots grouped by OS with subplots for each Python/PsychoPy version."""
    print("ℹ️  Creating per-OS plots with subplots...")

    for os_name, os_group in df.groupby("os"):
        variants_in_os = os_group["variant"].unique()
        n_variants = len(variants_in_os)
        
        if n_variants == 0:
            continue
            
        n_cols = min(3, n_variants)
        n_rows = (n_variants + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f"Installation Duration: {os_name}", fontsize=16, y=0.98)
        
        if n_variants == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if n_variants > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, variant in enumerate(variants_in_os):
            ax = axes[idx]
            grp = os_group[os_group["variant"] == variant].copy()
            grp = grp.sort_values("started").reset_index(drop=True)
            
            if grp.empty:
                ax.set_visible(False)
                continue
            
            run_ids, run_to_idx, run_dates_mapping, x_positions = setup_plot_data(grp)
            release_markers = prepare_release_markers(releases, run_dates_mapping, max_releases=5)

            ax.scatter(x_positions, grp["duration_m"].values, marker="o", alpha=0.7, 
                      s=30, zorder=3, label='Individual Runs')
            
            add_run_average_line(ax, grp, run_to_idx)
            
            python_ver = grp["python_version"].iloc[0] if len(grp) > 0 else "Unknown"
            psychopy_ver = grp["psychopy_version"].iloc[0] if len(grp) > 0 else "Unknown"
            title = f"Python {python_ver}, PsychoPy {psychopy_ver}"
            setup_axis_formatting(ax, run_ids, title, max_ticks=10)
            
            add_release_markers(ax, release_markers)
            
            if idx == 0:
                ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        
        for idx in range(n_variants, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        path = PLOTS_DIR / f"{sanitize_filename(os_name)}_subplots.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ OS '{os_name}': {n_variants} variants → {path}")


def create_combined_plot(df, releases):
    """Create a combined plot with all runs."""
    print("ℹ️  Creating combined plot...")

    df_all = df.sort_values("started").reset_index(drop=True)
    run_ids, run_to_idx, run_dates_mapping, x_all = setup_plot_data(df_all)
    
    release_markers = prepare_release_markers(releases, run_dates_mapping)

    fig, ax = plt.subplots(figsize=PlotConfig.COMBINED_PLOT_SIZE)
    
    variants = df_all["variant"].unique()
    
    # Create extended color palette for many variants
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    extra_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
                    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
                    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
    all_colors = base_colors + extra_colors
    color_cycle = cycle(all_colors)
    
    for variant in variants:
        mask = df_all["variant"] == variant
        x_variant = [x_all[j] for j in range(len(x_all)) if mask.iloc[j]]
        y_variant = [df_all["duration_m"].values[j] for j in range(len(x_all)) if mask.iloc[j]]
        ax.scatter(x_variant, y_variant, alpha=0.7, label=variant,
                   color=next(color_cycle), s=30, zorder=3)

    add_run_average_line(ax, df_all, run_to_idx)
    
    setup_axis_formatting(ax, run_ids, 
                         "All Runs: Setup Environment and Install (with GitHub Releases)", 
                         max_ticks=PlotConfig.MAX_TICKS_COMBINED)
    
    add_release_markers(ax, release_markers)

    handles, labels = ax.get_legend_handles_labels()
    if release_markers:
        release_line = Line2D([0], [0], color='red', linestyle='--', alpha=0.7)
        handles.append(release_line)
        labels.append('GitHub Releases')
    
    if len(handles) > 8:
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    else:
        ax.legend(handles, labels, loc='upper left', framealpha=0.9)
    
    fig.tight_layout()

    combined_path = PLOTS_DIR / "all_runs.png"
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    total_runs = len(df_all)
    span_runs = len(run_ids)
    releases_in_range = len(release_markers)
    print(f"✅ Combined plot: {total_runs} runs over {span_runs} unique run IDs, "
          f"{releases_in_range} releases → {combined_path}")


def create_averages_comparison_plot(df, releases):
    """Create subplots showing aggregated run averages by Python, PsychoPy, and OS categories."""
    print("ℹ️  Creating averages comparison plot with subplots...")

    df_all = df.sort_values("started").reset_index(drop=True)
    run_ids, run_to_idx, run_dates_mapping, _ = setup_plot_data(df_all)
    
    if not run_ids:
        print("⚠️  No data for averages comparison plot")
        return

    release_markers = prepare_release_markers(releases, run_dates_mapping, max_releases=5)

    # Create figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Define the three category types
    subplot_configs = [
        {
            'ax': axes[0],
            'title': 'Python Versions',
            'categories': [(f'Python {py_ver}', df[df['python_version'] == py_ver]) 
                          for py_ver in sorted(df['python_version'].unique())],
            'color_map': 'tab10'
        },
        {
            'ax': axes[1], 
            'title': 'PsychoPy Versions',
            'categories': [(f'PsychoPy {psychopy_ver}', df[df['psychopy_version'] == psychopy_ver])
                          for psychopy_ver in sorted(df['psychopy_version'].unique())],
            'color_map': 'Set1'
        },
        {
            'ax': axes[2],
            'title': 'Operating Systems', 
            'categories': [(f'{os_name}', df[df['os'] == os_name])
                          for os_name in sorted(df['os'].unique())],
            'color_map': 'Set2'
        }
    ]
    
    # Plot each subplot
    for subplot_config in subplot_configs:
        ax = subplot_config['ax']
        categories = subplot_config['categories']
        color_map = plt.colormaps[subplot_config['color_map']]
        
        # Add "All Runs" line to each subplot
        run_avg_all = calculate_run_average(df_all)
        if not run_avg_all.empty:
            ra_x_all, ra_y_all = [], []
            for _, row in run_avg_all.iterrows():
                run_id = row['run_id']
                if run_id in run_to_idx:
                    ra_x_all.append(run_to_idx[run_id])
                    ra_y_all.append(row['duration_m'])
            
            if ra_x_all and ra_y_all:
                ax.plot(ra_x_all, ra_y_all, color='black', linewidth=3, alpha=0.8, 
                       label='All Runs', zorder=5)
        
        # Add category-specific lines
        for i, (label, data) in enumerate(categories):
            if data.empty:
                continue
                
            color = color_map(i / max(1, len(categories) - 1))
            
            run_avg = calculate_run_average(data)
            if run_avg.empty:
                continue
            
            ra_x, ra_y = [], []
            for _, row in run_avg.iterrows():
                run_id = row['run_id']
                if run_id in run_to_idx:
                    ra_x.append(run_to_idx[run_id])
                    ra_y.append(row['duration_m'])
            
            if ra_x and ra_y:
                ax.plot(ra_x, ra_y, color=color, linewidth=2, alpha=0.8, 
                       label=label, zorder=4)

        # Format the subplot
        setup_axis_formatting(ax, run_ids, 
                             f"Installation Duration: {subplot_config['title']}", 
                             max_ticks=15)
        
        # Add release markers
        add_release_markers(ax, release_markers)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if release_markers and subplot_config['title'] == 'Python Versions':  # Only add to first subplot
            release_line = Line2D([0], [0], color='red', linestyle='--', alpha=0.7)
            handles.append(release_line)
            labels.append('GitHub Releases')
        
        # Sort legend so "All Runs" comes first
        sorted_items = sorted(zip(handles, labels), key=lambda x: (x[1] != 'All Runs', x[1]))
        handles, labels = zip(*sorted_items)
        
        ax.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=9)
    
    fig.tight_layout()

    averages_path = PLOTS_DIR / "averages_comparison.png"
    fig.savefig(averages_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Averages comparison plot with subplots → {averages_path}")



def create_installer_downloads_plot(releases):
    """Create a bar plot showing psychopy_linux_installer downloads per release."""
    print("ℹ️  Creating installer downloads plot...")
    
    release_data = []
    
    for release in releases:
        tag = release.get('tag_name', 'unknown')
        assets = release.get('assets', [])
        
        # Find the specific installer file downloads
        for asset in assets:
            if categorize_asset(asset['name']) == 'installer':
                asset_name = asset.get('name', 'unknown')
                downloads = asset.get('download_count', 0)
                
                # Skip entries with no downloads or no file name
                if downloads > 0 and asset_name and asset_name != 'unknown':
                    release_data.append({
                        'tag': tag,
                        'file': asset_name,
                        'downloads': downloads
                    })
    
    if not release_data:
        print("⚠️  No installer downloads found")
        return None
    
    # Sort by tag name (which should be chronological for version tags)
    release_data.sort(key=lambda x: x['tag'], reverse=True)
    
    # Limit to most recent releases for readability
    max_releases_to_show = 20
    if len(release_data) > max_releases_to_show:
        release_data = release_data[:max_releases_to_show]
    
    labels = [f"{r['tag']}\n({r['file']})" for r in release_data]
    downloads = [r['downloads'] for r in release_data]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(release_data) * 0.5)))
    
    bars = ax.barh(range(len(labels)), downloads, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Downloads', fontsize=12)
    ax.set_ylabel('Release (File)', fontsize=12)
    ax.set_title('psychopy_linux_installer Downloads per Release', fontsize=14, pad=20)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, downloads)):
        ax.text(count, i, f' {count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    path = PLOTS_DIR / "installer_downloads.png"
    fig.savefig(path, dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Installer downloads plot → {path}")
    return path


def create_wx_wheel_downloads_plot(releases):
    """Create a bar plot showing wx wheel downloads with full names including distro."""
    print("ℹ️  Creating wx wheel downloads plot...")
    
    wheel_downloads = defaultdict(int)
    
    for release in releases:
        assets = release.get('assets', [])
        
        for asset in assets:
            asset_name = asset.get('name', '')
            asset_type = categorize_asset(asset_name)
            
            # Only count wx wheels (ignore python wheels)
            if asset_type == 'wx_wheel':
                downloads = asset.get('download_count', 0)
                # Only include wheels with downloads and valid names
                if downloads > 0 and asset_name:
                    wheel_downloads[asset_name] += downloads
    
    if not wheel_downloads:
        print("⚠️  No wx wheel downloads found")
        return None
    
    # Sort by downloads (descending)
    sorted_wheels = sorted(
        wheel_downloads.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Limit for readability
    max_wheels_to_show = 30
    if len(sorted_wheels) > max_wheels_to_show:
        sorted_wheels = sorted_wheels[:max_wheels_to_show]
    
    wheel_names = [w[0] for w in sorted_wheels]
    downloads = [w[1] for w in sorted_wheels]
    
    fig, ax = plt.subplots(figsize=(14, max(10, len(wheel_names) * 0.4)))
    
    bars = ax.barh(range(len(wheel_names)), downloads, color='darkorange', alpha=0.7)
    ax.set_yticks(range(len(wheel_names)))
    ax.set_yticklabels(wheel_names, fontsize=8)
    ax.set_xlabel('Total Downloads (across all releases)', fontsize=12)
    ax.set_ylabel('Wheel File (includes wxPython version and distro)', fontsize=12)
    ax.set_title('Total wx Wheel Downloads', fontsize=14, pad=20)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, downloads)):
        ax.text(count, i, f' {count:,}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    path = PLOTS_DIR / "wx_wheel_downloads.png"
    fig.savefig(path, dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close(fig)
    
    total_downloads = sum(downloads)
    print(f"✅ Wx wheel downloads plot → {path}")
    print(f"   Total wx wheel downloads: {total_downloads:,}")
    return path


def create_download_summary_plot(releases):
    """Create a summary plot showing download distribution by asset type."""
    print("ℹ️  Creating download summary plot...")
    
    downloads_by_type = defaultdict(int)
    
    for release in releases:
        assets = release.get('assets', [])
        
        for asset in assets:
            asset_name = asset.get('name', '')
            downloads = asset.get('download_count', 0)
            asset_type = categorize_asset(asset_name)
            downloads_by_type[asset_type] += downloads
    
    if not downloads_by_type:
        print("⚠️  No download data found")
        return None
    
    # Create only bar chart (pie chart removed as requested)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    labels = []
    sizes = []
    colors = ['steelblue', 'darkorange', 'lightcoral', 'lightgray']
    
    for asset_type in ['installer', 'wx_wheel', 'python_wheel', 'other']:
        count = downloads_by_type.get(asset_type, 0)
        if count > 0:
            labels.append(asset_type.replace('_', ' ').title())
            sizes.append(count)
    
    # Bar chart
    if labels and sizes:
        bars = ax.bar(labels, sizes, color=colors[:len(labels)], alpha=0.7)
        ax.set_ylabel('Downloads', fontsize=12)
        ax.set_title('Total Downloads by Asset Type', fontsize=14, pad=20)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    path = PLOTS_DIR / "download_summary.png"
    fig.savefig(path, dpi=PlotConfig.DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Download summary plot → {path}")
    return path


def generate_readme(created_plots):
    """Generate README.md with the created plots."""
    print("ℹ️  Generating README.md...")
    
    readme_content = "# Output Data\n\n"
    
    for plot_info in created_plots:
        plot_name = plot_info['name']
        plot_path = plot_info['path']
        relative_path = f"duration_plots/{plot_path.name}"
        
        readme_content += f"## {plot_name}\n"
        readme_content += f"![{plot_name}]({relative_path})\n\n"
    
    readme_path = SCRIPT_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ README.md generated with {len(created_plots)} plots → {readme_path}")


def main():
    """Main plotting process."""
    print("📊 Starting plot generation...")

    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_data()
    if df is None:
        return

    releases = fetch_releases()

    created_plots = []

    # Generate all plots
    create_variant_plots(df, releases)
    create_combined_plot(df, releases)
    create_averages_comparison_plot(df, releases)
    
    # New visualizations
    dist_path = create_distribution_plot(df)
    heatmap_path = create_performance_heatmap(df)
    
    # Fetch releases with download data
    releases_with_assets = fetch_releases_with_assets()
    
    # Generate download plots
    installer_downloads_path = create_installer_downloads_plot(releases_with_assets)
    wx_downloads_path = create_wx_wheel_downloads_plot(releases_with_assets)
    summary_path = create_download_summary_plot(releases_with_assets)

    # Register plots for README
    combined_path = PLOTS_DIR / "all_runs.png"
    created_plots.append({
        'name': "All Runs",
        'path': combined_path
    })
    
    averages_path = PLOTS_DIR / "averages_comparison.png"
    created_plots.append({
        'name': "Averages Comparison",
        'path': averages_path
    })
    
    created_plots.append({
        'name': "Distribution Box Plots",
        'path': dist_path
    })
    
    created_plots.append({
        'name': "Performance Heatmap",
        'path': heatmap_path
    })

    for os_name, os_group in df.groupby("os"):
        variants_in_os = os_group["variant"].unique()
        if len(variants_in_os) > 0:
            plot_path = PLOTS_DIR / f"{sanitize_filename(os_name)}_subplots.png"
            created_plots.append({
                'name': f"{os_name} Subplots",
                'path': plot_path
            })
    
    # Add download plots to README
    if installer_downloads_path:
        created_plots.append({
            'name': "Installer Downloads per Release",
            'path': installer_downloads_path
        })
    
    if wx_downloads_path:
        created_plots.append({
            'name': "Total wx Wheel Downloads",
            'path': wx_downloads_path
        })
    
    if summary_path:
        created_plots.append({
            'name': "Download Summary by Asset Type",
            'path': summary_path
        })

    generate_readme(created_plots)

    print("✅ Plot generation complete!")


if __name__ == "__main__":
    main()
