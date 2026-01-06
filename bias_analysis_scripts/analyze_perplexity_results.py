#!/usr/bin/env python3
"""
This script analyzes and visualizes the perplexity results from the dialect perplexity calculation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize dialect perplexity results.")
    parser.add_argument("--input_file", type=str, default="../output_datasets/dialect_perplexity_results.csv",
                        help="Path to the input CSV file containing perplexity results.")
    parser.add_argument("--output_dir", type=str, default="../results/dialect_perplexity_analysis",
                        help="Directory to save analysis results and visualizations.")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load results
    print(f"Loading results from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Basic statistics
    print("Calculating statistics...")
    stats = {
        'count': len(df),
        'mean_aae_perplexity': df['aae_perplexity'].mean(),
        'mean_sae_perplexity': df['sae_perplexity'].mean(),
        'median_aae_perplexity': df['aae_perplexity'].median(),
        'median_sae_perplexity': df['sae_perplexity'].median(),
        'std_aae_perplexity': df['aae_perplexity'].std(),
        'std_sae_perplexity': df['sae_perplexity'].std(),
        'mean_difference': df['perplexity_difference'].mean(),
        'median_difference': df['perplexity_difference'].median(),
        'mean_ratio': df['perplexity_ratio'].mean(),
        'median_ratio': df['perplexity_ratio'].median(),
    }
    
    # Print summary
    print("\nSummary Statistics:")
    print(f"Total examples: {stats['count']}")
    print(f"Mean AAE perplexity: {stats['mean_aae_perplexity']:.4f}")
    print(f"Mean SAE perplexity: {stats['mean_sae_perplexity']:.4f}")
    print(f"Median AAE perplexity: {stats['median_aae_perplexity']:.4f}")
    print(f"Median SAE perplexity: {stats['median_sae_perplexity']:.4f}")
    print(f"Mean difference (AAE - SAE): {stats['mean_difference']:.4f}")
    print(f"Median difference (AAE - SAE): {stats['median_difference']:.4f}")
    print(f"Mean ratio (AAE / SAE): {stats['mean_ratio']:.4f}")
    print(f"Median ratio (AAE / SAE): {stats['median_ratio']:.4f}")
    
    # Save summary to file
    with open(os.path.join(args.output_dir, 'summary_statistics.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Distribution of perplexities
    plt.subplot(2, 2, 1)
    sns.histplot(data=df[['aae_perplexity', 'sae_perplexity']], 
                 bins=30, kde=True, alpha=0.7)
    plt.title('Distribution of Perplexity Scores')
    plt.xlabel('Perplexity')
    plt.ylabel('Frequency')
    plt.legend(['AAE', 'SAE'])
    
    # 2. Boxplot comparison
    plt.subplot(2, 2, 2)
    df_melted = pd.melt(df, value_vars=['aae_perplexity', 'sae_perplexity'], 
                         var_name='Dialect', value_name='Perplexity')
    sns.boxplot(x='Dialect', y='Perplexity', data=df_melted)
    plt.title('Perplexity by Dialect')
    plt.ylabel('Perplexity')
    
    # 3. Scatter plot of AAE vs SAE perplexity
    plt.subplot(2, 2, 3)
    plt.scatter(df['sae_perplexity'], df['aae_perplexity'], alpha=0.5)
    # Add diagonal line
    max_val = max(df['aae_perplexity'].max(), df['sae_perplexity'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    plt.title('AAE vs. SAE Perplexity')
    plt.xlabel('SAE Perplexity')
    plt.ylabel('AAE Perplexity')
    
    # 4. Distribution of differences
    plt.subplot(2, 2, 4)
    sns.histplot(df['perplexity_difference'], bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Distribution of Perplexity Differences (AAE - SAE)')
    plt.xlabel('Perplexity Difference')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'perplexity_analysis.png'), dpi=300)
    
    # Create additional analysis figures
    
    # 5. Perplexity ratio distribution
    plt.figure(figsize=(10, 6))
    # Filter out extreme ratios for better visualization
    filtered_ratios = df[df['perplexity_ratio'] < 5]['perplexity_ratio']
    sns.histplot(filtered_ratios, bins=30, kde=True)
    plt.axvline(x=1, color='r', linestyle='--', alpha=0.7)
    plt.title('Distribution of Perplexity Ratios (AAE / SAE)')
    plt.xlabel('Perplexity Ratio')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'perplexity_ratio_distribution.png'), dpi=300)
    
    # 6. Find examples with highest and lowest differences
    df_sorted_by_diff = df.sort_values('perplexity_difference', ascending=False)
    highest_diff = df_sorted_by_diff.head(10)
    lowest_diff = df_sorted_by_diff.tail(10)
    
    # Save examples to CSV
    highest_diff.to_csv(os.path.join(args.output_dir, 'highest_perplexity_differences.csv'), index=False)
    lowest_diff.to_csv(os.path.join(args.output_dir, 'lowest_perplexity_differences.csv'), index=False)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
