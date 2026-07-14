import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve, f1_score
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from cell_line_utils import check_cell_lines_in_chip, validate_file_not_empty

@dataclass
class SampleConfig:
    """Configuration for a sample."""
    label: str
    sample: str
    ground_truth_file: str


def find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Find the threshold that gives the best F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    return best_threshold, f1_scores[best_idx]


def calculate_metrics_by_tf(sample_configs: List[SampleConfig], dfs: List[pl.DataFrame], 
                           aligned_chip_data_dir: Optional[str] = None,
                           cell_line_mapping: Optional[Dict[str, str]] = None) -> pl.DataFrame:
    """
    Calculate metrics for each transcription factor using the check_cell_lines_in_chip function.
    
    Args:
        sample_configs: List of SampleConfig objects
        dfs: List of corresponding DataFrames with predictions
        aligned_chip_data_dir: Directory with aligned ATAC data
        cell_line_mapping: Dictionary mapping cell line keys to names
        
    Returns:
        Polars DataFrame with TF, cell_line_count, and f1_score
    """
    results = []
    
    for config, df in zip(sample_configs, dfs):
        # Get true labels and predictions
        y_true = np.array(df["targets"])
        y_score = np.array(df["probabilities"])
        
        # Find best threshold for F1 score
        best_threshold, _ = find_best_f1_threshold(y_true, y_score)
        
        # Calculate F1 score using best threshold
        y_pred = (y_score >= best_threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        # Get directory of the ground truth file to count cell lines
        directory = os.path.dirname(config.ground_truth_file)
        
        # Count valid cell lines using check_cell_lines_in_chip if we have alignment info
        if aligned_chip_data_dir and cell_line_mapping:
            cell_line_count = check_cell_lines_in_chip(
                directory,
                cell_line_mapping,
                aligned_chip_data_dir
            )
        else:
            # Fall back to counting files if we don't have alignment info
            cell_line_files = {
                f.split("_")[0]: os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if not f.startswith('.') and f.endswith(".bed")
            }
            cell_line_count = sum(1 for f in cell_line_files.values() 
                                if validate_file_not_empty(f))
        
        # Add to results
        results.append({
            "transcription_factor": config.label,
            "cell_line": config.sample,
            "cell_line_count": cell_line_count,
            "f1_score": f1,
            "positive_count": df.filter(pl.col("targets") == 1).height,
            "negative_count": df.filter(pl.col("targets") == 0).height,
            "total_count": df.height
        })
    
    # Convert to Polars DataFrame
    return pl.from_dicts(results)


def plot_tf_metrics(metrics_df: pl.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create visualizations showing relationship between cell line count and F1 score.
    
    Args:
        metrics_df: Polars DataFrame with metrics data
        save_path: Optional path to save the plot
    """
    # Convert to pandas for easier plotting with seaborn
    pdf = metrics_df.to_pandas()
    
    # Sort the dataframe by cell_line_count for the bar chart
    pdf_sorted = pdf.sort_values('cell_line_count')
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create a colormap based on cell line count - from light to dark
    norm = plt.Normalize(pdf['cell_line_count'].min(), pdf['cell_line_count'].max())
    cmap = plt.cm.Blues  # Use Blues colormap which goes from light to dark
    colors = cmap(norm(pdf['cell_line_count']))
    colors_sorted = cmap(norm(pdf_sorted['cell_line_count']))
    
    # Plot 1: Cell Line Count vs F1 Score
    sns.scatterplot(
        x="cell_line_count", 
        y="f1_score", 
        data=pdf, 
        s=150, 
        alpha=0.8, 
        palette=colors,
        ax=axes[0]
    )
    
    # Add a linear trend line through the points
    x_data = pdf['cell_line_count']
    y_data = pdf['f1_score']
    # Calculate linear fit coefficients (using linear scale)
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    # Create x range for line (using linear scale)
    x_min, x_max = x_data.min() * 0.9, x_data.max() * 1.1
    x_range = np.linspace(x_min, x_max, 100)
    # Plot the trend line
    axes[0].plot(x_range, p(x_range), '--', color='black', linewidth=1.5)
    
    # Add labels for each point
    for i, row in pdf.iterrows():
        axes[0].annotate(
            row['transcription_factor'], 
            (row['cell_line_count'], row['f1_score']),
            xytext=(7, 7), 
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )
    
    # F1 score range (without log scale on x-axis)
    axes[0].set_ylim([0, 1])
    
    axes[0].set_xlabel('Number of Cell Lines (log scale)', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('Relationship Between Number of Cell Lines and F1 Score', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparing F1 scores
    bar_colors = colors
    bars = axes[1].bar(
        pdf['transcription_factor'], 
        pdf['f1_score'], 
        alpha=0.8,
        color=bar_colors
    )
    
    # Add cell line count as text on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'Lines: {pdf.iloc[i]["cell_line_count"]}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    axes[1].set_xlabel('Transcription Factor', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score by Transcription Factor', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_tf_metrics_detailed(metrics_df: pl.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create a detailed dashboard of transcription factor metrics.
    
    Args:
        metrics_df: Polars DataFrame with metrics data
        save_path: Optional path to save the plot
    """
    # Convert to pandas for easier plotting with seaborn
    pdf = metrics_df.to_pandas()
    
    # Sort the dataframe by cell_line_count for the bar chart
    pdf_sorted = pdf.sort_values('cell_line_count')
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3)
    
    # Create a colormap based on cell line count - from light to dark
    norm = plt.Normalize(pdf['cell_line_count'].min(), pdf['cell_line_count'].max())
    cmap = plt.cm.Blues  # Use Blues colormap which goes from light to dark
    colors = cmap(norm(pdf['cell_line_count']))
    colors_sorted = cmap(norm(pdf_sorted['cell_line_count']))
    
    # Plot 1: Cell Line Count vs F1 Score (top left)
    sns.scatterplot(
        x="cell_line_count", 
        y="f1_score", 
        data=pdf, 
        s=200, 
        alpha=0.8, 
        palette=colors,
        ax=axes[0, 0]
    )
    
    # Add a linear trend line through the points
    x_data = pdf['cell_line_count']
    y_data = pdf['f1_score']
    # Calculate linear fit coefficients (using linear scale)
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    # Create x range for line (using linear scale)
    x_min, x_max = x_data.min() * 0.9, x_data.max() * 1.1
    x_range = np.linspace(x_min, x_max, 100)
    # Plot the trend line
    axes[0, 0].plot(x_range, p(x_range), '--', color='black', linewidth=1.5)
    
    # Add labels for each point
    for i, row in pdf.iterrows():
        axes[0, 0].annotate(
            row['transcription_factor'], 
            (row['cell_line_count'], row['f1_score']),
            xytext=(8, 8), 
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )
    
    # F1 score range (without log scale on x-axis)
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 0].set_xlabel('Number of Cell Lines (log scale)', fontsize=14)
    axes[0, 0].set_ylabel('F1 Score', fontsize=14)
    axes[0, 0].set_title('Cell Line Count vs F1 Score', fontsize=16)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparing F1 scores ordered by cell line count (top right)
    bars = axes[0, 1].bar(
        pdf_sorted['transcription_factor'], 
        pdf_sorted['f1_score'], 
        alpha=0.8,
        color=colors_sorted
    )
    
    # Add cell line count as text on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'Lines: {pdf_sorted.iloc[i]["cell_line_count"]}',
            ha='center', 
            va='bottom',
            fontsize=11
        )
    
    # Add a colorbar to show the relationship between color and cell line count
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0, 1], orientation='vertical', pad=0.1)
    cbar.set_label('Number of Cell Lines', fontsize=12)
    
    axes[0, 1].set_xlabel('Transcription Factor', fontsize=14)
    axes[0, 1].set_ylabel('F1 Score', fontsize=14)
    axes[0, 1].set_title('F1 Score by Transcription Factor', fontsize=16)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Positive/Negative sample distribution (bottom left)
    # Sort by cell line count
    sorted_tfs = pdf_sorted['transcription_factor'].tolist()
    data_for_stacked = pdf.set_index('transcription_factor').loc[sorted_tfs][['positive_count', 'negative_count']]
    data_for_stacked.plot(
        kind='bar', 
        stacked=True, 
        ax=axes[1, 0],
        color=['#2ecc71', '#e74c3c'],
        alpha=0.8
    )
    
    # Add percentage text
    for i, tf in enumerate(pdf['transcription_factor']):
        pos = pdf.iloc[i]['positive_count']
        neg = pdf.iloc[i]['negative_count']
        total = pos + neg
        percent_pos = (pos / total) * 100
        
        # Add text for positive percentage
        axes[1, 0].text(
            i, 
            pos/2,
            f'{percent_pos:.1f}%',
            ha='center',
            va='center',
            fontsize=11,
            color='white',
            fontweight='bold'
        )
        
        # Add text for negative percentage
        axes[1, 0].text(
            i, 
            pos + neg/2,
            f'{100-percent_pos:.1f}%',
            ha='center',
            va='center',
            fontsize=11,
            color='white',
            fontweight='bold'
        )
    
    axes[1, 0].set_xlabel('Transcription Factor', fontsize=14)
    axes[1, 0].set_ylabel('Sample Count', fontsize=14)
    axes[1, 0].set_title('Positive/Negative Sample Distribution', fontsize=16)
    axes[1, 0].legend(['Positive Samples', 'Negative Samples'])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Metrics table (bottom right)
    # Remove the axis
    axes[1, 1].axis('off')
    
    # Create a table with sorted data
    table_data = [
        ['TF', 'F1', 'Pos Samples', 'Neg Samples', 'Cell Lines'],
    ]
    
    for i, row in pdf_sorted.iterrows():
        table_data.append([
            row['transcription_factor'],
            f"{row['f1_score']:.3f}",
            f"{row['positive_count']}",
            f"{row['negative_count']}",
            f"{row['cell_line_count']}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.18, 0.18, 0.18, 0.18, 0.18]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the header row
    for j in range(len(table_data[0])):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#3498db')
    
    # Alternate row colors for better readability
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('#e6e6e6')
    
    axes[1, 1].set_title('Transcription Factor Metrics Summary', fontsize=16)
    
    # Add a main title to the figure
    fig.suptitle('Transcription Factor Performance Analysis Dashboard', fontsize=20, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 
    