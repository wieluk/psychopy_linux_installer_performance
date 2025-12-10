"""Shared configuration for the installer speed analysis scripts."""
from pathlib import Path

# Directory structure
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PLOTS_DIR = SCRIPT_DIR / "duration_plots"

# File paths
CSV_FILE = DATA_DIR / "durations.csv"
JSON_CACHE = DATA_DIR / "runs.json"

# GitHub repository to track
REPO = "wieluk/psychopy_linux_installer"

# GitHub Actions step name to track
STEP_NAME = "Setup environment and install"


class PlotConfig:
    """Configuration for plot generation."""
    # Tick densities for different plot types
    MAX_TICKS_COMBINED = 25
    MAX_TICKS_SUBPLOT = 15
    MAX_TICKS_AVERAGES = 20
    
    # Release label configuration
    RELEASE_FONT_SIZE = 8
    RELEASE_FONT_SIZE_SMALL = 7
    RELEASE_FONT_SIZE_TINY = 6
    RELEASE_HEIGHT_LEVELS = 8  # Number of vertical positions for labels
    MAX_RELEASES_TO_SHOW = 10  # Maximum number of releases to display
    
    # Plot dimensions
    COMBINED_PLOT_SIZE = (16, 9)
    SUBPLOT_SIZE_PER_PANEL = (5, 4)
    AVERAGES_PLOT_SIZE = (24, 8)
    DISTRIBUTION_PLOT_SIZE = (24, 8)
    HEATMAP_PLOT_SIZE = (12, 10)
    
    # Export settings
    DPI = 150


class FetchConfig:
    """Configuration for data fetching."""
    BATCH_SIZE = 10
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 60  # seconds
    PER_PAGE = 100  # GitHub API results per page


# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
