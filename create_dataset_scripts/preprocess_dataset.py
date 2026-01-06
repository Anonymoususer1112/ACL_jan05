"""
Script to preprocess the dataset by removing extra quotation marks.
"""

import os
import sys
import pandas as pd
import logging
import argparse
import html
import re
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("dataset_preprocessing.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def clean_quotation_marks(text):
    """
    Remove leading and trailing quotation marks from text.
    Also handles escaped quotes, complex nested quote patterns, and consecutive quotes.
    Decodes HTML entities like &lt; and &amp;
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text without problematic quotation marks and HTML entities
    """
    if not isinstance(text, str):
        return text
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # First pass: Deal with simple quote patterns at beginning and end
    # This handles the case where the entire text is wrapped in quotes
    if text.startswith('"') and text.endswith('"') and len(text) > 1:
        text = text[1:-1]
    
    # Second pass: Handle multiple consecutive quotes at beginning and end
    # Count and remove consecutive quotes at the beginning
    start_quotes = 0
    for char in text:
        if char == '"':
            start_quotes += 1
        else:
            break
    
    if start_quotes > 0:
        text = text[start_quotes:]
    
    # Count and remove consecutive quotes at the end
    end_quotes = 0
    for char in reversed(text):
        if char == '"':
            end_quotes += 1
        else:
            break
    
    if end_quotes > 0:
        text = text[:-end_quotes]
    
    # Handle escaped quotes
    text = text.replace('\\"', '"')
    text = text.replace('\\"', '"')  # Handle double escaping if present
    
    # Fix backslashes that might be part of internal quoted phrases
    text = text.replace("\\", "")
    
    # Replace consecutive quotes with a single quote throughout the text
    # This handles cases like "" or """ appearing in the middle of text
    text = re.sub(r'"{2,}', '"', text)
    
    # Handle complex quote patterns that might remain
    # Fix common CSV parsing artifacts where quotes are doubled
    text = re.sub(r'""([^"]+)""', r'"\1"', text)

    # Decode HTML entities (like &lt; and &amp;)
    text = html.unescape(text)
    
    # Final clean-up: ensure we don't have dangling quotes at beginning/end
    text = text.strip('"')
    
    return text

def preprocess_dataset(input_path, output_path):
    """
    Preprocess the dataset by removing extra quotation marks and HTML entities.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    try:
        # Load the dataset
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Process the text column
        logger.info("Cleaning text data - removing quotation marks and HTML entities")
        if 'text' in df.columns:
            df['text'] = df['text'].apply(clean_quotation_marks)
        
        # Process other columns if they exist
        for col in ['sae_text', 'aae_text', 'original_text']:
            if col in df.columns:
                df[col] = df[col].apply(clean_quotation_marks)
                
        # Save the processed dataset
        logger.info(f"Saving preprocessed dataset to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} records")
        
        return True
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset by removing extra quotation marks")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file")
    args = parser.parse_args()
    
    if preprocess_dataset(args.input, args.output):
        logger.info("Preprocessing completed successfully")
    else:
        logger.error("Preprocessing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()