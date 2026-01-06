#!/usr/bin/env python3
"""
Script to compare sentiment analysis results between Claude Batch API and GPT-4.1 Batch API.
This helps in evaluating the consistency of the two models for dialectic preference bias research.
"""

import os
import sys
import csv
import pandas as pd
import logging
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClaudeBatchInterface, GPT41BatchInterface

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("model_comparison.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

def compare_models_on_dataset(input_file: str, output_file: str, text_column: str, num_samples: int = None):
    """
    Compare sentiment analysis results between Claude and GPT-4.1 on a dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        text_column: Name of column containing text data
        num_samples: Optional number of samples to process (for testing)
    """
    # Load dataset
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows")
    
    # Sample if requested
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
        logger.info(f"Sampled {num_samples} rows for processing")
    
    # Check if text column exists
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found in dataset. Available columns: {', '.join(df.columns)}")
        return
    
    # Extract texts to analyze
    texts = df[text_column].tolist()
    logger.info(f"Extracted {len(texts)} texts for sentiment analysis")
    
    # Initialize model interfaces
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    gpt41_api_key = os.environ.get("OPENAI_API_KEY")
    
    claude = ClaudeBatchInterface(api_key=claude_api_key)
    gpt41 = GPT41BatchInterface(api_key=gpt41_api_key)
    
    # Perform batch sentiment analysis with Claude
    logger.info("Starting batch sentiment analysis with Claude")
    claude_sentiments = claude.batch_get_sentiment(texts)
    logger.info(f"Completed Claude sentiment analysis for {len(claude_sentiments)} texts")
    
    # Perform batch sentiment analysis with GPT-4.1
    logger.info("Starting batch sentiment analysis with GPT-4.1")
    gpt41_sentiments = gpt41.batch_get_sentiment(texts)
    logger.info(f"Completed GPT-4.1 sentiment analysis for {len(gpt41_sentiments)} texts")
    
    # Add sentiment results to DataFrame
    df['claude_sentiment'] = [s['sentiment'] for s in claude_sentiments]
    df['claude_score'] = [s['score'] for s in claude_sentiments]
    df['gpt41_sentiment'] = [s['sentiment'] for s in gpt41_sentiments]
    df['gpt41_score'] = [s['score'] for s in gpt41_sentiments]
    
    # Add agreement column
    df['models_agree'] = df['claude_sentiment'] == df['gpt41_sentiment']
    
    # Save results
    logger.info(f"Saving comparison results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Calculate and display agreement metrics
    agreement_rate = df['models_agree'].mean() * 100
    logger.info(f"Model agreement rate: {agreement_rate:.2f}%")
    
    # Display sentiment distribution
    claude_sentiment_counts = df['claude_sentiment'].value_counts()
    gpt41_sentiment_counts = df['gpt41_sentiment'].value_counts()
    logger.info(f"Claude sentiment distribution: {dict(claude_sentiment_counts)}")
    logger.info(f"GPT-4.1 sentiment distribution: {dict(gpt41_sentiment_counts)}")
    
    # Check for errors
    claude_error_count = sum(1 for s in df['claude_sentiment'] if s == 'ERROR')
    gpt41_error_count = sum(1 for s in df['gpt41_sentiment'] if s == 'ERROR')
    if claude_error_count > 0:
        logger.warning(f"{claude_error_count} records had errors during Claude processing")
    if gpt41_error_count > 0:
        logger.warning(f"{gpt41_error_count} records had errors during GPT-4.1 processing")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Compare Claude and GPT-4.1 sentiment analysis")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file")
    parser.add_argument("--text-column", "-t", default="text", help="Column name with text to analyze")
    parser.add_argument("--samples", "-s", type=int, help="Number of samples to process (optional)")
    args = parser.parse_args()
    
    compare_models_on_dataset(args.input, args.output, args.text_column, args.samples)

if __name__ == "__main__":
    main()
