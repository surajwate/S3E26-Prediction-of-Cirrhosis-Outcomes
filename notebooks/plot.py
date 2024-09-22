import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_custom_config import SEABORN_RC, SEABORN_PALETTE

def plot_count_with_percentages(data, variable, title="Count Plot", xlabel=None, ylabel="Count", palette="Set2"):
    """
    Function to create a vertical count plot with percentage labels for binary or categorical variables.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the variable to plot.
    variable (str): The name of the variable/column in the dataset to plot.
    title (str): The title of the plot. Default is "Count Plot".
    xlabel (str): The label for the x-axis. If None, it uses the variable name.
    ylabel (str): The label for the y-axis. Default is "Count".
    palette (str): The color palette for the plot. Default is "Set2".
    
    Returns:
    None: Displays the plot.
    """
    
    # Set default xlabel to the variable if not provided
    if xlabel is None:
        xlabel = variable
    
    # Set plot size and aesthetics
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", rc=SEABORN_RC)
    sns.set_context("notebook", font_scale=1.2)

    # Dynamically set the color palette based on the number of unique values in the variable
    palette = sns.color_palette(palette, n_colors=data[variable].nunique())
    
    # Create the vertical countplot with hue for the variable and custom color palette
    ax = sns.countplot(data=data, x=variable, hue=variable, palette=palette, legend=False)
    
    # Calculate total number of observations
    total = len(data)
    
    # Add percentage labels to each bar
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, 
                    (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom', fontsize=12, color='black')
    
    # Set title and axis labels with larger font sizes
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Remove top and right spines for a cleaner look
    sns.despine()
    
    # Display the plot
    plt.tight_layout()
    plt.show()

'''
Example usage:

plot_count_with_percentages(train, 'faulty', title="Distribution of Faulty vs Non-Faulty Steel Plates", xlabel="Fault Type", ylabel="Count")

'''