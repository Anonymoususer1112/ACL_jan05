#!/usr/bin/env python3
"""
Example script for performing sentiment analysis on a dataset using Claude Batch API.
"""

import os
import sys
import pandas as pd
import logging
import argparse
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClaudeBatchInterface

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("claude_sentiment_analysis.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

def analyze_dataset_sentiment(input_file: str, output_file: str, text_column: str, num_samples: int = None):
    """
    Analyze sentiment of texts in a dataset using Claude Batch API.
    
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
    
    # Initialize Claude batch interface
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = ClaudeBatchInterface(api_key=api_key)
    
    # Perform batch sentiment analysis
    logger.info("Starting batch sentiment analysis")
    sentiments = model.batch_get_sentiment(texts)
    logger.info(f"Completed sentiment analysis for {len(sentiments)} texts")
    
    # Add sentiment results to DataFrame
    df['sentiment'] = [s['sentiment'] for s in sentiments]
    # df['sentiment_score'] = [s['score'] for s in sentiments]
    # df['raw_response'] = [s['raw_response'] for s in sentiments]
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Display sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
    
    # Check for errors
    error_count = sum(1 for s in df['sentiment'] if s == 'ERROR')
    if error_count > 0:
        logger.warning(f"{error_count} records had errors during processing")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment using Claude Batch API")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file")
    parser.add_argument("--text-column", "-t", default="text", help="Column name with text to analyze")
    parser.add_argument("--samples", "-s", type=int, help="Number of samples to process (optional)")
    args = parser.parse_args()
    
    analyze_dataset_sentiment(args.input, args.output, args.text_column, args.samples)

if __name__ == "__main__":
    main()
