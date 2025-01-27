import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
from seaborn_custom_config import SEABORN_RC, SEABORN_PALETTE

sns.set_theme(style="whitegrid", rc=SEABORN_RC)
sns.set_context("notebook", font_scale=1.2)


def is_jupyter_notebook():
    """
    Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in a Jupyter notebook or JupyterLab, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if "ZMQInteractiveShell" in shell:
            return True  # Jupyter notebook or JupyterLab
        elif "TerminalInteractiveShell" in shell:
            return False  # Terminal running IPython
        else:
            return False  # Other type (likely terminal)
    except NameError:
        return False  # Probably standard Python interpreter


def categorical_feature(
    df, feature, target, show_plot=True, save_plot=False, plot_dir="./plots"
):
    """
    Calculates and visualizes the distribution of a categorical feature with respect to a target variable.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The name of the categorical feature.
        target (str): The name of the target variable.
        show_plot (bool): Whether to display the plot. Default is True.
        save_plot (bool): Whether to save the plot to a file. Default is False.
        plot_dir (str): Directory to save the plot if save_plot is True. Default is './plots'.

    Returns:
        pd.DataFrame: A DataFrame containing the distribution of the feature, including the total count, total percentage,
                      percentages for each target class relative to the total, and percentages of each target class within
                      the feature category.

    Raises:
        ValueError: If the feature or target column is not found in the DataFrame.
    """
    # Validate input columns
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")

    # Calculate total counts and percentages
    total_counts = df[feature].value_counts()
    total_percentages = df[feature].value_counts(normalize=True) * 100

    # Calculate counts and percentages for each target class
    counts_per_class = df.groupby([feature, target]).size().unstack(fill_value=0)
    percentages_per_class_of_total = (
        counts_per_class.div(len(df)) * 100
    )  # Percentages relative to total
    percentages_within_feature = (
        counts_per_class.div(counts_per_class.sum(axis=1), axis=0) * 100
    )  # Percentages within feature

    # Combine all into a single DataFrame
    category_distribution = pd.DataFrame(
        {
            "Total Count": total_counts,
            "Total Percentage": total_percentages,
        }
    )

    # Add percentages per class relative to total
    for class_value in counts_per_class.columns:
        category_distribution[f"{class_value} of Total (%)"] = (
            percentages_per_class_of_total[class_value]
        )

    # Add percentages per class within feature category
    for class_value in counts_per_class.columns:
        category_distribution[f"{class_value} within {feature} (%)"] = (
            percentages_within_feature[class_value]
        )

    # Sort the DataFrame by total count
    category_distribution = category_distribution.sort_values(
        "Total Count", ascending=False
    )

    # Plotting
    plt.figure(figsize=(12, 6))

    # Dynamically set the color palette based on the number of target classes
    palette = sns.color_palette(SEABORN_PALETTE, n_colors=len(df[target].unique()))

    sns.countplot(
        data=df,
        x=feature,
        hue=target,
        order=category_distribution.index,
        palette=palette, alpha=0.8
    )
    plt.title(f"Distribution of '{feature}' by '{target}'")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)

    # Show or save the plot based on parameters
    if show_plot:
        plt.show()

    if save_plot:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_filename = f"{feature}-{target}-distribution.png"
        plt.savefig(os.path.join(plot_dir, plot_filename), bbox_inches="tight")
        print(f"Plot saved to {os.path.join(plot_dir, plot_filename)}")

    plt.close()  # Close the figure to free memory

    return category_distribution


def numerical_feature(
    df,
    feature,
    target=None,
    figsize=(15, 6),
    bins="sturges",
    show_plot=True,
    save_plot=False,
    plot_dir="./plots",
):
    """
    Analyzes a numerical feature in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - feature (str): The name of the numerical feature to analyze.
    - target (str, optional): The name of the target column for grouping the analysis. Default is None.
    - figsize (tuple, optional): The size of the figure. Default is (15, 6).
    - bins (int or str, optional): The number of bins for the histogram or the method to calculate it. Options are:
        - 'sturges' (default)
        - 'rice'
        - 'scott'
        - 'fd' (Freedman-Diaconis)
        - an integer specifying the number of bins
    - show_plot (bool, optional): Whether to display the plot. Default is True.
    - save_plot (bool, optional): Whether to save the plot as a PNG file. Default is False.
    - plot_dir (str, optional): Directory to save the plot if save_plot is True. Default is './plots'.

    Returns:
    - outliers_df (pd.DataFrame): A DataFrame containing the percentage of outliers in the data.
    - summary_df (pd.DataFrame): A DataFrame containing the overall statistics, lower outliers statistics, and upper outliers statistics.
    """

    # Validate input columns
    if feature not in df.columns:
        raise ValueError(
            f"Feature '{feature}' not found in DataFrame columns: {list(df.columns)}"
        )
    if target and target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found in DataFrame columns: {list(df.columns)}"
        )

    # Calculate the number of bins if a method is provided
    if isinstance(bins, str):
        if bins == "sturges":
            bins = int(np.ceil(np.log2(len(df[feature].dropna())) + 1))
        elif bins == "rice":
            bins = int(np.ceil(2 * len(df[feature].dropna()) ** (1 / 3)))
        elif bins == "scott":
            bin_width = 3.5 * df[feature].std() * len(df[feature].dropna()) ** (-1 / 3)
            bins = int(np.ceil((df[feature].max() - df[feature].min()) / bin_width))
        elif bins == "fd":  # Freedman-Diaconis
            bin_width = (
                2
                * (df[feature].quantile(0.75) - df[feature].quantile(0.25))
                * len(df[feature].dropna()) ** (-1 / 3)
            )
            bins = int(np.ceil((df[feature].max() - df[feature].min()) / bin_width))
        else:
            raise ValueError(f"Unknown binning method: '{bins}'")
    elif not isinstance(bins, int):
        raise TypeError(
            "Bins must be an integer or one of the following strings: 'sturges', 'rice', 'scott', 'fd'."
        )

    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # First plot: Histogram with KDE

    # Determine the number of unique classes in the target column, or relevant variable
    num_classes = len(df[target].unique()) if target else len(df[feature].unique())

    # Dynamically set the color palette based on the correct number of target classes
    palette = sns.color_palette(SEABORN_PALETTE, n_colors=num_classes)


    sns.histplot(
        data=df,
        x=feature,
        hue=target,
        bins=bins,
        kde=True,
        ax=ax[0],
        palette=palette,
        element="bars",
    )
    ax[0].set_title(f"Distribution of '{feature}'")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Second plot: Boxplot of the feature by the target (if provided)

    # Dynamically set the color palette based on the number of unique values in the feature
    palette = sns.color_palette(SEABORN_PALETTE, n_colors=len(df[feature].unique()))

    if target:
        sns.boxplot(
            data=df,
            x=feature,
            y=target,
            orient="h",
            ax=ax[1],
            palette="coolwarm",
            hue=target,
            legend=False, saturation=0.8
        )
        ax[1].set_title(f"Box Plot of '{feature}' by '{target}'")
        ax[1].set_ylabel(target)
    else:
        sns.boxplot(data=df, x=feature, orient="h", ax=ax[1], palette="coolwarm")
        ax[1].set_title(f"Box Plot of '{feature}'")
        ax[1].set_ylabel("")

    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)
    ax[1].set_xlabel(feature)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show or save the plot based on parameters
    if show_plot:
        plt.show()

    if save_plot:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_filename = (
            f"{feature}_distribution.png"
            if not target
            else f"{feature}_by_{target}_distribution.png"
        )
        plt.savefig(os.path.join(plot_dir, plot_filename), bbox_inches="tight")
        print(f"Plot saved to {os.path.join(plot_dir, plot_filename)}")

    plt.close()  # Close the figure to free memory

    # Calculate overall statistics
    overall_summary = df[feature].describe().to_frame().T
    overall_summary.index = [f"{feature}_Overall"]

    # Calculate the lower and upper bounds for outliers using IQR method
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify lower and upper bound outliers
    lower_outliers = df[df[feature] < lower_bound]
    upper_outliers = df[df[feature] > upper_bound]

    # Get descriptive statistics for lower and upper outliers
    lower_outliers_summary = lower_outliers[feature].describe().to_frame().T
    lower_outliers_summary.index = [f"{feature}_Lower_Outliers"]

    upper_outliers_summary = upper_outliers[feature].describe().to_frame().T
    upper_outliers_summary.index = [f"{feature}_Upper_Outliers"]

    # Combine overall statistics with lower and upper outlier statistics
    summary_df = pd.concat(
        [overall_summary, lower_outliers_summary, upper_outliers_summary]
    )

    # Calculate the percentage of outliers in the data
    total_outliers = len(lower_outliers) + len(upper_outliers)
    outlier_percentage = (total_outliers / len(df[feature].dropna())) * 100
    lower_outliers_percentage = (len(lower_outliers) / len(df[feature].dropna())) * 100
    upper_outliers_percentage = (len(upper_outliers) / len(df[feature].dropna())) * 100

    outliers_df = pd.DataFrame(
        {
            "Total Outliers": [total_outliers],
            "Outlier Percentage (%)": [outlier_percentage],
            "Lower Outliers": [len(lower_outliers)],
            "Lower Outliers Percentage (%)": [lower_outliers_percentage],
            "Upper Outliers": [len(upper_outliers)],
            "Upper Outliers Percentage (%)": [upper_outliers_percentage],
        }
    )

    return outliers_df, summary_df


def missing_values(df, include_all=False, visualize=False, figsize=(10, 6)):
    """
    Generates a summary of missing values in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame to analyze.
        include_all (bool, optional): If True, includes columns with zero missing values in the summary. Default is False.
        visualize (bool, optional): If True, displays a bar plot of missing percentages. Default is False.
        figsize (tuple, optional): Figure size for the visualization. Default is (10, 6).

    Returns:
        pd.DataFrame: A DataFrame containing the count and percentage of missing values,
                      along with the data type of each column.
    """
    # Calculate missing values count and percentage
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df) * 100).round(2)
    data_types = df.dtypes

    # Combine into a DataFrame
    missing_summary = pd.DataFrame(
        {
            "Missing Count": missing_count,
            "Missing Percentage (%)": missing_percentage,
            "Data Type": data_types,
        }
    )

    # Filter out columns with no missing values if include_all is False
    if not include_all:
        missing_summary = missing_summary[missing_summary["Missing Count"] > 0]

    # Sort by Missing Percentage in descending order
    missing_summary = missing_summary.sort_values(
        by="Missing Percentage (%)", ascending=False
    )

    # Visualize missing data if requested
    if visualize and not missing_summary.empty:
        plt.figure(figsize=figsize)
        sns.barplot(
            x=missing_summary.index, y="Missing Percentage (%)", data=missing_summary
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Missing Data Percentage by Column")
        plt.ylabel("Missing Percentage (%)")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.show()

    return missing_summary