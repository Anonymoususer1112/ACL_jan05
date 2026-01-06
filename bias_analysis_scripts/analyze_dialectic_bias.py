"""
Script to analyze dialect preference bias using the Dialectic Group Invariance (DGI) metric.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import argparse
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bias_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_sentiment_datasets(aae_path: str, sae_path: str, aae_from_sae_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three sentiment datasets for analysis.
    
    Args:
        aae_path: Path to AAE sentiment dataset
        sae_path: Path to SAE sentiment dataset
        aae_from_sae_path: Path to AAE-from-SAE sentiment dataset
        
    Returns:
        Tuple of three DataFrames (aae_df, sae_df, aae_from_sae_df)
    """
    try:
        aae_df = pd.read_csv(aae_path)
        logger.info(f"Loaded AAE dataset with {len(aae_df)} records")
        
        sae_df = pd.read_csv(sae_path)
        logger.info(f"Loaded SAE dataset with {len(sae_df)} records")
        
        aae_from_sae_df = pd.read_csv(aae_from_sae_path)
        logger.info(f"Loaded AAE-from-SAE dataset with {len(aae_from_sae_df)} records")
        
        return aae_df, sae_df, aae_from_sae_df
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def merge_datasets(aae_df: pd.DataFrame, sae_df: pd.DataFrame, aae_from_sae_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three datasets into a single DataFrame for analysis.
    
    Args:
        aae_df: DataFrame with AAE sentiment results
        sae_df: DataFrame with SAE sentiment results
        aae_from_sae_df: DataFrame with AAE-from-SAE sentiment results
        
    Returns:
        Merged DataFrame
    """
    try:
        # Extract relevant columns
        aae_subset = aae_df[['aae_text', 'sentiment']].rename(
            columns={'text': 'aae_text', 'sentiment': 'aae_sentiment'})
        
        sae_subset = sae_df[['sae_text', 'sentiment']].rename(
            columns={'sentiment': 'sae_sentiment', 'score': 'sae_score'})
        
        aae_from_sae_subset = aae_from_sae_df[['aae_from_sae_text', 'sentiment']].rename(
            columns={'sentiment': 'aae_from_sae_sentiment'})
        
        # Merge datasets (assuming they have the same order)
        # For a more robust solution, you would need a common key to join on
        merged_df = pd.concat([aae_subset, sae_subset, aae_from_sae_subset], axis=1)
        logger.info(f"Created merged dataset with {len(merged_df)} records")
        
        return merged_df
    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        raise

def calculate_dgi_metrics(merged_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Dialectic Group Invariance (DGI) metrics.
    
    Args:
        merged_df: DataFrame with merged sentiment results
        
    Returns:
        Dictionary with DGI metrics
    """
    try:
        # Calculate DGI between AAE and SAE
        aae_sae_matches = (merged_df['aae_sentiment'] == merged_df['sae_sentiment']).sum()
        dgi_aae_sae = aae_sae_matches / len(merged_df)
        logger.info(f"DGI(AAE, SAE) = {dgi_aae_sae:.4f}")
        
        # Calculate DGI between AAE and AAE-from-SAE
        aae_aae_from_sae_matches = (merged_df['aae_sentiment'] == merged_df['aae_from_sae_sentiment']).sum()
        dgi_aae_aae_from_sae = aae_aae_from_sae_matches / len(merged_df)
        logger.info(f"DGI(AAE, AAE-from-SAE) = {dgi_aae_aae_from_sae:.4f}")
        
        # Calculate DGI between SAE and AAE-from-SAE
        sae_aae_from_sae_matches = (merged_df['sae_sentiment'] == merged_df['aae_from_sae_sentiment']).sum()
        dgi_sae_aae_from_sae = sae_aae_from_sae_matches / len(merged_df)
        logger.info(f"DGI(SAE, AAE-from-SAE) = {dgi_sae_aae_from_sae:.4f}")
        
        # Calculate three-way DGI (AAE = SAE = AAE-from-SAE)
        three_way_matches = ((merged_df['aae_sentiment'] == merged_df['sae_sentiment']) & 
                             (merged_df['sae_sentiment'] == merged_df['aae_from_sae_sentiment'])).sum()
        dgi_three_way = three_way_matches / len(merged_df)
        logger.info(f"DGI(AAE, SAE, AAE-from-SAE) = {dgi_three_way:.4f}")
        
        # Calculate metrics by sentiment category
        sentiment_categories = ['positive', 'negative', 'neutral']
        dgi_by_sentiment = {}
        
        for sentiment in sentiment_categories:
            # Filter by AAE sentiment
            sentiment_df = merged_df[merged_df['aae_sentiment'] == sentiment]
            
            if len(sentiment_df) > 0:
                # Calculate DGI for this sentiment
                aae_sae_matches = (sentiment_df['aae_sentiment'] == sentiment_df['sae_sentiment']).sum()
                dgi_aae_sae_sentiment = aae_sae_matches / len(sentiment_df)
                
                aae_aae_from_sae_matches = (sentiment_df['aae_sentiment'] == sentiment_df['aae_from_sae_sentiment']).sum()
                dgi_aae_aae_from_sae_sentiment = aae_aae_from_sae_matches / len(sentiment_df)
                
                three_way_matches = ((sentiment_df['aae_sentiment'] == sentiment_df['sae_sentiment']) & 
                                    (sentiment_df['sae_sentiment'] == sentiment_df['aae_from_sae_sentiment'])).sum()
                dgi_three_way_sentiment = three_way_matches / len(sentiment_df)
                
                dgi_by_sentiment[sentiment] = {
                    'count': len(sentiment_df),
                    'dgi_aae_sae': dgi_aae_sae_sentiment,
                    'dgi_aae_aae_from_sae': dgi_aae_aae_from_sae_sentiment,
                    'dgi_three_way': dgi_three_way_sentiment
                }
                
                logger.info(f"DGI for {sentiment} sentiment (n={len(sentiment_df)}):")
                logger.info(f"  DGI(AAE, SAE) = {dgi_aae_sae_sentiment:.4f}")
                logger.info(f"  DGI(AAE, AAE-from-SAE) = {dgi_aae_aae_from_sae_sentiment:.4f}")
                logger.info(f"  DGI(AAE, SAE, AAE-from-SAE) = {dgi_three_way_sentiment:.4f}")
        
        # Return all metrics
        return {
            'dgi_aae_sae': dgi_aae_sae,
            'dgi_aae_aae_from_sae': dgi_aae_aae_from_sae,
            'dgi_sae_aae_from_sae': dgi_sae_aae_from_sae,
            'dgi_three_way': dgi_three_way,
            'dgi_by_sentiment': dgi_by_sentiment,
            'sample_size': len(merged_df)
        }
    
    except Exception as e:
        logger.error(f"Error calculating DGI metrics: {str(e)}")
        raise

def analyze_sentiment_shifts(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in sentiment shifts between dialects.
    
    Args:
        merged_df: DataFrame with merged sentiment results
        
    Returns:
        Dictionary with shift analysis results
    """
    try:
        # Count cases where AAE and SAE sentiments differ
        shifts = merged_df[merged_df['aae_sentiment'] != merged_df['sae_sentiment']]
        shift_count = len(shifts)
        shift_percentage = (shift_count / len(merged_df)) * 100
        
        logger.info(f"Found {shift_count} sentiment shifts ({shift_percentage:.2f}% of samples)")
        
        # Analyze direction of shifts
        shift_patterns = {}
        sentiment_categories = ['positive', 'negative', 'neutral']
        
        for aae_sentiment in sentiment_categories:
            for sae_sentiment in sentiment_categories:
                if aae_sentiment != sae_sentiment:
                    pattern = f"{aae_sentiment}_to_{sae_sentiment}"
                    count = len(merged_df[(merged_df['aae_sentiment'] == aae_sentiment) & 
                                         (merged_df['sae_sentiment'] == sae_sentiment)])
                    if count > 0:
                        shift_patterns[pattern] = {
                            'count': count,
                            'percentage': (count / len(merged_df)) * 100,
                            'percentage_of_shifts': (count / shift_count) * 100 if shift_count > 0 else 0
                        }
                        logger.info(f"  {pattern}: {count} instances ({shift_patterns[pattern]['percentage']:.2f}% of total)")
        
        # Direction of bias
        positive_to_negative = shift_patterns.get('positive_to_negative', {}).get('count', 0)
        negative_to_positive = shift_patterns.get('negative_to_positive', {}).get('count', 0)
        
        positive_bias = negative_to_positive - positive_to_negative
        negative_bias = positive_to_negative - negative_to_positive
        
        if positive_bias > 0:
            logger.info(f"Overall bias towards more positive sentiment in SAE: +{positive_bias} instances")
        elif negative_bias > 0:
            logger.info(f"Overall bias towards more negative sentiment in SAE: +{negative_bias} instances")
        else:
            logger.info("No clear directional bias in sentiment shifts")
        
        # Return analysis results
        return {
            'shift_count': shift_count,
            'shift_percentage': shift_percentage,
            'shift_patterns': shift_patterns,
            'positive_bias': positive_bias,
            'negative_bias': negative_bias
        }
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment shifts: {str(e)}")
        raise

def generate_visualizations(merged_df: pd.DataFrame, model_name: str, output_dir: str):
    """
    Generate visualizations of bias analysis results.
    
    Args:
        merged_df: DataFrame with merged sentiment results
        model_name: Name of the model being analyzed
        output_dir: Directory to save visualizations
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Sentiment distribution by dialect
        plt.figure(figsize=(12, 6))
        
        # Get sentiment counts for each dialect
        aae_counts = merged_df['aae_sentiment'].value_counts().reindex(['positive', 'negative', 'neutral']).fillna(0)
        sae_counts = merged_df['sae_sentiment'].value_counts().reindex(['positive', 'negative', 'neutral']).fillna(0)
        aae_from_sae_counts = merged_df['aae_from_sae_sentiment'].value_counts().reindex(['positive', 'negative', 'neutral']).fillna(0)
        
        # Set up bar positions
        bar_width = 0.25
        r1 = np.arange(len(aae_counts))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        plt.bar(r1, aae_counts, width=bar_width, label='AAE', color='skyblue')
        plt.bar(r2, sae_counts, width=bar_width, label='SAE', color='lightcoral')
        plt.bar(r3, aae_from_sae_counts, width=bar_width, label='AAE-from-SAE', color='lightgreen')
        
        # Add labels and legend
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title(f'Sentiment Distribution by Dialect ({model_name})')
        plt.xticks([r + bar_width for r in range(len(aae_counts))], ['Positive', 'Negative', 'Neutral'])
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_sentiment_distribution.png'))
        plt.close()
        
        # 2. Sentiment agreement heatmap
        plt.figure(figsize=(10, 8))
        
        # Create a contingency table
        contingency = pd.crosstab(merged_df['aae_sentiment'], merged_df['sae_sentiment'], normalize='all') * 100
        
        # Create heatmap
        sns.heatmap(contingency, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage of Total'})
        plt.title(f'AAE vs. SAE Sentiment Agreement ({model_name})')
        plt.xlabel('SAE Sentiment')
        plt.ylabel('AAE Sentiment')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_sentiment_agreement.png'))
        plt.close()
        
        # 3. Dialectic Group Invariance bar chart
        plt.figure(figsize=(10, 6))
        
        # Calculate DGI values
        dgi_aae_sae = (merged_df['aae_sentiment'] == merged_df['sae_sentiment']).mean()
        dgi_aae_aae_from_sae = (merged_df['aae_sentiment'] == merged_df['aae_from_sae_sentiment']).mean()
        dgi_sae_aae_from_sae = (merged_df['sae_sentiment'] == merged_df['aae_from_sae_sentiment']).mean()
        dgi_three_way = ((merged_df['aae_sentiment'] == merged_df['sae_sentiment']) & 
                         (merged_df['sae_sentiment'] == merged_df['aae_from_sae_sentiment'])).mean()
        
        # Create bar chart
        dgi_values = [dgi_aae_sae, dgi_aae_aae_from_sae, dgi_sae_aae_from_sae, dgi_three_way]
        dgi_labels = ['AAE vs SAE', 'AAE vs AAE-from-SAE', 'SAE vs AAE-from-SAE', 'Three-way Agreement']
        
        plt.bar(dgi_labels, dgi_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.xlabel('Comparison')
        plt.ylabel('Dialectic Group Invariance (DGI)')
        plt.title(f'Dialectic Group Invariance Metrics ({model_name})')
        plt.ylim([0, 1])
        
        # Add value labels
        for i, v in enumerate(dgi_values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_dgi_metrics.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Analyze dialect preference bias using DGI metrics")
    parser.add_argument("--aae", required=True, help="Path to AAE sentiment dataset")
    parser.add_argument("--sae", required=True, help="Path to SAE sentiment dataset")
    parser.add_argument("--aae-from-sae", required=True, help="Path to AAE-from-SAE sentiment dataset")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file for results")
    parser.add_argument("--model", "-m", required=True, help="Name of the model being analyzed")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualizations of results")
    args = parser.parse_args()
    
    try:
        # Load datasets
        aae_df, sae_df, aae_from_sae_df = load_sentiment_datasets(
            args.aae, 
            args.sae, 
            args.aae_from_sae
        )
        
        # Merge datasets
        merged_df = merge_datasets(aae_df, sae_df, aae_from_sae_df)
        
        # Calculate DGI metrics
        dgi_metrics = calculate_dgi_metrics(merged_df)
        
        # Analyze sentiment shifts
        shift_analysis = analyze_sentiment_shifts(merged_df)
        
        # Combine results
        results = {
            'model_name': args.model,
            'dgi_metrics': dgi_metrics,
            'shift_analysis': shift_analysis,
            'dataset_sizes': {
                'aae': len(aae_df),
                'sae': len(sae_df),
                'aae_from_sae': len(aae_from_sae_df),
                'merged': len(merged_df)
            }
        }
        
        # Add sentiment distribution statistics
        raw_aae = merged_df['aae_sentiment'].value_counts().to_dict()
        aae_counts = { label: int(count) for label, count in raw_aae.items() }

        raw_sae = merged_df['sae_sentiment'].value_counts().to_dict()
        sae_counts = { label: int(count) for label, count in raw_sae.items() }

        raw_aae_from_sae = merged_df['aae_from_sae_sentiment'].value_counts().to_dict()
        aae_from_sae_counts = { label: int(count) for label, count in raw_aae_from_sae.items() }

        results['sentiment_distribution'] = {
            'aae': aae_counts,
            'sae': sae_counts,
            'aae_from_sae': aae_from_sae_counts
        }

        
        # Save results to JSON
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Generate visualizations if requested
        if args.visualize:
            vis_output_dir = os.path.join(output_dir, 'visualizations')
            generate_visualizations(merged_df, args.model, vis_output_dir)
            logger.info(f"Visualizations saved to {vis_output_dir}")
        
    except Exception as e:
        logger.error(f"Error during bias analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()