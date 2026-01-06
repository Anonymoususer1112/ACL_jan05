#!/usr/bin/env python3
"""
Script to clean the sentiment analysis results by removing the raw_response column
and renaming the 'text' column to 'aae_text'.
"""

import os
import sys
import pandas as pd
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sentiment_cleaning.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def clean_sentiment_file(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Clean a sentiment analysis results file by removing the raw_response column
    and renaming 'text' column to 'aae_text'.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the cleaned CSV file (defaults to input path with '_cleaned' suffix)
    """
    # Default output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"
    
    try:
        # Load the dataset
        logger.info(f"Loading sentiment results from {input_path}")
        df = pd.read_csv(input_path)
        original_rows = len(df)
        logger.info(f"Loaded {original_rows} rows")
        
        # Check if the file has the expected columns
        if 'text' not in df.columns:
            logger.error(f"Input file missing required 'text' column")
            sys.exit(1)
        
        if 'sentiment' not in df.columns:
            logger.error(f"Input file missing required 'sentiment' column")
            sys.exit(1)
            
        # Create a new DataFrame with only the needed columns
        if 'score' in df.columns:
            cleaned_df = pd.DataFrame({
                'aae_text': df['text'],
                'sentiment': df['sentiment']
            })
        else:
            cleaned_df = pd.DataFrame({
                'aae_text': df['text'],
                'sentiment': df['sentiment']
            })
            
        # Save the cleaned DataFrame
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned sentiment results to {output_path}")
        logger.info(f"Columns in cleaned output: {', '.join(cleaned_df.columns)}")
        
        # Sentiment distribution stats
        sentiment_counts = cleaned_df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
        
    except Exception as e:
        logger.error(f"Error cleaning sentiment file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Clean sentiment analysis results file")
    parser.add_argument("--input", "-i", required=True, 
                        help="Path to input sentiment results CSV file")
    parser.add_argument("--output", "-o", 
                        help="Path to save the cleaned output (default: input_cleaned.csv)")
    args = parser.parse_args()
    
    clean_sentiment_file(args.input, args.output)

if __name__ == "__main__":
    main()
