"""
Script to compare bias analysis results across multiple models.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("model_comparison.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load results from multiple model analyses.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        Dictionary mapping model names to their results
    """
    results = {}
    
    try:
        # List all JSON files in the results directory
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        
        if not result_files:
            logger.warning(f"No result files found in {results_dir}")
            return results
        
        logger.info(f"Found {len(result_files)} result files")
        
        # Load each result file
        for result_file in result_files:
            file_path = os.path.join(results_dir, result_file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Use model name from the file if available, otherwise use filename
                model_name = data.get('model_name', os.path.splitext(result_file)[0])
                results[model_name] = data
                
                logger.info(f"Loaded results for {model_name}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise

def compare_dgi_metrics(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare DGI metrics across models.
    
    Args:
        model_results: Dictionary mapping model names to their results
        
    Returns:
        DataFrame with comparative DGI metrics
    """
    try:
        comparison = {
            'Model': [],
            'DGI(AAE, SAE)': [],
            'DGI(AAE, AAE-from-SAE)': [],
            'DGI(SAE, AAE-from-SAE)': [],
            'DGI(3-way)': [],
            'Sample Size': []
        }
        
        for model_name, results in model_results.items():
            dgi_metrics = results.get('dgi_metrics', {})
            
            comparison['Model'].append(model_name)
            comparison['DGI(AAE, SAE)'].append(dgi_metrics.get('dgi_aae_sae', 0))
            comparison['DGI(AAE, AAE-from-SAE)'].append(dgi_metrics.get('dgi_aae_aae_from_sae', 0))
            comparison['DGI(SAE, AAE-from-SAE)'].append(dgi_metrics.get('dgi_sae_aae_from_sae', 0))
            comparison['DGI(3-way)'].append(dgi_metrics.get('dgi_three_way', 0))
            comparison['Sample Size'].append(dgi_metrics.get('sample_size', 0))
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('DGI(AAE, SAE)', ascending=True)  # Sort by the main DGI metric
        
        return df
    
    except Exception as e:
        logger.error(f"Error comparing DGI metrics: {str(e)}")
        raise

def compare_sentiment_shifts(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare sentiment shift patterns across models.
    
    Args:
        model_results: Dictionary mapping model names to their results
        
    Returns:
        DataFrame with comparative shift metrics
    """
    try:
        comparison = {
            'Model': [],
            'Shift Percentage': [],
            'Positive Bias': [],
            'Negative Bias': [],
            'pos→neg': [],
            'neg→pos': [],
            'pos→neu': [],
            'neu→pos': [],
            'neg→neu': [],
            'neu→neg': []
        }
        
        for model_name, results in model_results.items():
            shift_analysis = results.get('shift_analysis', {})
            shift_patterns = shift_analysis.get('shift_patterns', {})
            
            comparison['Model'].append(model_name)
            comparison['Shift Percentage'].append(shift_analysis.get('shift_percentage', 0))
            comparison['Positive Bias'].append(shift_analysis.get('positive_bias', 0))
            comparison['Negative Bias'].append(shift_analysis.get('negative_bias', 0))
            
            # Add specific shift patterns
            comparison['pos→neg'].append(shift_patterns.get('positive_to_negative', {}).get('percentage', 0))
            comparison['neg→pos'].append(shift_patterns.get('negative_to_positive', {}).get('percentage', 0))
            comparison['pos→neu'].append(shift_patterns.get('positive_to_neutral', {}).get('percentage', 0))
            comparison['neu→pos'].append(shift_patterns.get('neutral_to_positive', {}).get('percentage', 0))
            comparison['neg→neu'].append(shift_patterns.get('negative_to_neutral', {}).get('percentage', 0))
            comparison['neu→neg'].append(shift_patterns.get('neutral_to_negative', {}).get('percentage', 0))
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Shift Percentage', ascending=False)  # Sort by shift percentage
        
        return df
    
    except Exception as e:
        logger.error(f"Error comparing sentiment shifts: {str(e)}")
        raise

def generate_comparative_visualizations(model_results: Dict[str, Dict], output_dir: str):
    """
    Generate visualizations comparing results across models.
    
    Args:
        model_results: Dictionary mapping model names to their results
        output_dir: Directory to save visualizations
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. DGI metrics comparison
        dgi_df = compare_dgi_metrics(model_results)
        
        plt.figure(figsize=(12, 8))
        
        # Reshape DataFrame for easier plotting with seaborn
        dgi_plot_df = dgi_df.melt(id_vars=['Model', 'Sample Size'], 
                                  value_vars=['DGI(AAE, SAE)', 'DGI(AAE, AAE-from-SAE)', 
                                             'DGI(SAE, AAE-from-SAE)', 'DGI(3-way)'],
                                  var_name='Metric', value_name='DGI Value')
        
        # Create grouped bar chart
        g = sns.catplot(x='Model', y='DGI Value', hue='Metric', data=dgi_plot_df, kind='bar', height=6, aspect=1.5)
        
        # Customize the plot
        g.set_xticklabels(rotation=45, ha='right')
        plt.title('DGI Metrics Comparison Across Models')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'dgi_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Shift percentage comparison
        shift_df = compare_sentiment_shifts(model_results)
        
        plt.figure(figsize=(10, 6))
        
        # Create bar chart of shift percentages
        sns.barplot(x='Model', y='Shift Percentage', data=shift_df)
        plt.title('Sentiment Shift Percentage by Model')
        plt.xlabel('Model')
        plt.ylabel('Percentage of Samples with Sentiment Shift')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(shift_df['Shift Percentage']):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shift_percentage_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Bias direction comparison
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar chart for bias direction
        bias_data = {
            'Model': shift_df['Model'],
            'Positive Bias': shift_df['Positive Bias'],
            'Negative Bias': shift_df['Negative Bias']
        }
        bias_df = pd.DataFrame(bias_data)
        bias_plot_df = bias_df.melt(id_vars=['Model'], value_vars=['Positive Bias', 'Negative Bias'],
                                   var_name='Bias Direction', value_name='Bias Strength')
        
        sns.barplot(x='Model', y='Bias Strength', hue='Bias Direction', data=bias_plot_df)
        plt.title('Dialect Bias Direction by Model')
        plt.xlabel('Model')
        plt.ylabel('Bias Strength (# of instances)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'bias_direction_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Shift pattern heatmap
        shift_patterns_df = shift_df[['Model', 'pos→neg', 'neg→pos', 'pos→neu', 'neu→pos', 'neg→neu', 'neu→neg']]
        shift_patterns_df = shift_patterns_df.set_index('Model')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(shift_patterns_df, annot=True, fmt='.1f', cmap='Blues')
        plt.title('Sentiment Shift Patterns by Model (% of Total Samples)')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'shift_patterns_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating comparative visualizations: {str(e)}")
        raise

def generate_readme_content(model_results: Dict[str, Dict]) -> str:
    """
    Generate markdown content for the README file.
    
    Args:
        model_results: Dictionary mapping model names to their results
        
    Returns:
        README content as string
    """
    try:
        # Create DGI metrics table
        dgi_df = compare_dgi_metrics(model_results)
        dgi_table = dgi_df.to_markdown(index=False, floatfmt='.4f')
        
        # Create shift metrics table
        shift_df = compare_sentiment_shifts(model_results)
        # Only include key columns for the table
        shift_table_df = shift_df[['Model', 'Shift Percentage', 'Positive Bias', 'Negative Bias']]
        shift_table = shift_table_df.to_markdown(index=False, floatfmt='.2f')
        
        # Generate content
        content = f"""# Dialectic Preference Bias in Large Language Models

## Project Overview

This research investigates Dialectic Preference Bias in various Large Language Models. We define Dialectic Preference Bias as the phenomenon when LLMs output reflects or promotes unfair preferences or prejudices towards particular dialects or linguistic variations of a language. Such bias may lead the model to favor certain ways of speaking or writing, which can disadvantage speakers of marginalized dialects and perpetuate social biases and inequalities.

## Methodology

Our analysis uses a comparative approach across different language models:

1. We start with a dataset of tweets in African American English (AAE)
2. Translate the AAE tweets to Standard American English (SAE)
3. Translate the SAE tweets back to AAE
4. Obtain sentiment classifications for all three versions
5. Calculate bias metrics to assess dialect preference bias

## Key Findings

### Dialectic Group Invariance (DGI) Metrics

The DGI metric quantifies consistency in sentiment classification across dialects. Higher values indicate less bias.

{dgi_table}

### Sentiment Shift Patterns

This table shows the percentage of samples where sentiment changed between dialects and the direction of bias.

{shift_table}

* **Positive Bias**: Higher values indicate a tendency to assign more positive sentiment to SAE compared to AAE
* **Negative Bias**: Higher values indicate a tendency to assign more negative sentiment to SAE compared to AAE

## Visualizations

See the `/bias_analysis_scripts/results/visualizations` directory for detailed visualizations of the results.

## Running the Analysis

To run the bias analysis on your own data:

```
# Example command to analyze a specific model's results
python bias_analysis_scripts/analyze_dialectic_bias.py \
  --aae output_datasets/MODEL_AAE_sentiment.csv \
  --sae output_datasets/MODEL_SAE_sentiment.csv \
  --aae-from-sae output_datasets/MODEL_AAE_from_SAE_sentiment.csv \
  --output bias_analysis_scripts/results/MODEL_results.json \
  --model MODEL_NAME \
  --visualize

# To compare results across models
python bias_analysis_scripts/compare_models.py \
  --results-dir bias_analysis_scripts/results \
  --output bias_analysis_scripts/results/comparative_analysis \
  --visualize
```

## Conclusion

Our research reveals varying degrees of dialectic preference bias across the evaluated LLMs. The DGI metrics and sentiment shift patterns provide quantitative evidence of this bias, showing that these models do not treat different dialects equally when performing sentiment analysis tasks.

These findings highlight the importance of continued research and improvement in making language models more equitable across different linguistic varieties and dialects.
"""
        
        return content
    
    except Exception as e:
        logger.error(f"Error generating README content: {str(e)}")
        return f"# Dialectic Preference Bias\n\nError generating content: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Compare bias analysis results across models")
    parser.add_argument("--results-dir", required=True, help="Directory containing result JSON files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for comparative results")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate comparative visualizations")
    parser.add_argument("--update-readme", "-r", action="store_true", help="Update main README.md with results")
    args = parser.parse_args()
    
    try:
        # Load results from all models
        model_results = load_results(args.results_dir)
        
        if not model_results:
            logger.error("No valid results found to compare")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Generate DataFrame comparisons
        dgi_comparison = compare_dgi_metrics(model_results)
        shift_comparison = compare_sentiment_shifts(model_results)
        
        # Save comparison DataFrames
        dgi_comparison.to_csv(os.path.join(args.output, 'dgi_comparison.csv'), index=False)
        shift_comparison.to_csv(os.path.join(args.output, 'shift_comparison.csv'), index=False)
        
        # Generate comparative visualizations if requested
        if args.visualize:
            vis_output_dir = os.path.join(args.output, 'visualizations')
            generate_comparative_visualizations(model_results, vis_output_dir)
            logger.info(f"Comparative visualizations saved to {vis_output_dir}")
        
        # Update README if requested
        if args.update_readme:
            readme_content = generate_readme_content(model_results)
            
            # Get path to repo root (assuming this script is in bias_analysis_scripts)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            readme_path = os.path.join(repo_root, 'README.md')
            
            with open(readme_path, 'w') as f:
                f.write(readme_content)
                
            logger.info(f"Updated README at {readme_path}")
        
        logger.info(f"Comparative analysis complete. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error during comparative analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()