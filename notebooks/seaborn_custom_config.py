# seaborn_custom_config.py

# Custom rc settings for Seaborn plots to match your website's theme
SEABORN_RC = {
    "axes.facecolor": "#fdf6e3",      # Axes background color
    "figure.facecolor": "#ece4d4",    # Figure background color 
    "axes.edgecolor": "#333",         # Axes border color
    "axes.labelcolor": "#333",        # Axes label color
    "xtick.color": "#333",            # X-axis tick color
    "ytick.color": "#333",            # Y-axis tick color
    "grid.color": "#e0e0e0",          # Grid line color
    "text.color": "#333",             # Text color
    "lines.color": "#1A6FB0",         # Default line color
    "patch.edgecolor": "none",     # Highlight border color, use #c58301 for orange
    "font.family": "serif",           # Default font family
    "font.serif": [
        "EB Garamond",   # Preferred font, matches the web style
        "Garamond",      # Fallback if EB Garamond is not available
        "Times New Roman",  # Common serif alternative
        "serif"          # Generic serif fallback
    ],
    # Font size settings
    "font.size": 12,                # Default font size for all elements
    "axes.titlesize": 14,           # Font size of plot titles
    "axes.labelsize": 12,           # Font size of axis labels
    "xtick.labelsize": 10,          # Font size of x-axis tick labels
    "ytick.labelsize": 10,          # Font size of y-axis tick labels
    "legend.fontsize": 11,          # Font size of legend text
    "figure.titlesize": 16,         # Font size of the figure title
}

# SEABORN_PALETTE: Custom palette to match your website's theme
SEABORN_PALETTE = [
    "#1A6FB0",  # Primary color: A deep blue shade, used for primary emphasis. Ideal for main data points or important categories.
    "#c58301",  # Secondary color: A strong orange/yellow shade. It provides a warm contrast, suitable for secondary data or highlighting key differences.
    "#d4a373",  # Accent color: A soft, earthy beige tone. Good for adding a subtle highlight or softening the palette when too many bold colors are present.
    "#b0c4de",  # Soft blue color: A light, desaturated blue, often used for backgrounds or non-intrusive data. Great for calm, neutral visuals.
    "#8c564b",  # Muted red/brown for contrast: A muted, earthy red-brown, which adds depth and contrast without overwhelming the plot. Ideal for error or anomaly indication.
    "#2ca02c",  # Green for additional variation: A bright green, providing a fresh and distinct contrast to the warmer and cooler tones above. Useful for success indicators or positive outcomes.
]

