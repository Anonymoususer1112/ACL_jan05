"""
Script to check the size of a JSONL file that would be created for batch processing.
This helps determine if we can process the entire dataset in one batch.
"""

import os
import sys
import json
import pandas as pd
import tempfile
import argparse
import logging
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_batch_request_file_for_sentiment(texts, model="gpt-4.1-2025-04-14", output_path=None):
    """Create a JSONL file for batch sentiment analysis and measure its size.
    
    Args:
        texts: List of texts to process
        model: Model ID to use
        output_path: Path to save the JSONL file (if None, uses a temp file)
        
    Returns:
        Tuple of (file path, file size in MB)
    """
    if output_path:
        # Use the provided path
        path = output_path
        file_mode = 'w'
    else:
        # Use a temporary file
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        file_mode = 'w'
    
    prompt_template = """
    Please analyze the sentiment of the following text and respond with exactly one word: 
    either 'positive', 'negative', or 'neutral'.

    Text: "{text}"
    
    Sentiment:
    """
    
    try:
        with open(path, file_mode) as f:
            for i, text in enumerate(tqdm(texts, desc="Creating batch file")):
                custom_id = f"request-{i}"
                
                # Format the text with the template
                formatted_text = prompt_template.format(text=text)
                    
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": formatted_text}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.3
                    }
                }
                f.write(json.dumps(request) + '\n')
        
        # Get file size in MB
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        return path, file_size_mb
    except Exception as e:
        logger.error(f"Error creating batch request file: {str(e)}")
        if not output_path and 'path' in locals():
            try:
                os.unlink(path)
            except:
                pass
        raise

def create_batch_request_file_for_translation(texts, model="gpt-4.1-2025-04-14", output_path=None):
    """Create a JSONL file for batch translation and measure its size.
    
    Args:
        texts: List of texts to process
        model: Model ID to use
        output_path: Path to save the JSONL file (if None, uses a temp file)
        
    Returns:
        Tuple of (file path, file size in MB)
    """
    if output_path:
        path = output_path
        file_mode = 'w'
    else:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        file_mode = 'w'
    
    prompt_template = """
    Translate the following tweet from African American English (AAE) to Standard American English (SAE).
    Preserve the meaning, tone, and intent of the original tweet.
    Only change dialectical features while maintaining the original message.
    Original tweet (AAE): "{text}"
    Standard American English translation:
    """
    
    try:
        with open(path, file_mode) as f:
            for i, text in enumerate(tqdm(texts, desc="Creating batch file")):
                custom_id = f"request-{i}"
                
                # Format the text with the template
                formatted_text = prompt_template.format(text=text)
                    
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": formatted_text}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.3
                    }
                }
                f.write(json.dumps(request) + '\n')
        
        # Get file size in MB
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        return path, file_size_mb
    except Exception as e:
        logger.error(f"Error creating batch request file: {str(e)}")
        if not output_path and 'path' in locals():
            try:
                os.unlink(path)
            except:
                pass
        raise

def estimate_batches_needed(file_size_mb, max_batch_size_mb=100):
    """Estimate how many batches would be needed given the file size.
    
    Args:
        file_size_mb: Size of the file in MB
        max_batch_size_mb: Maximum allowed batch size in MB
        
    Returns:
        Number of batches needed
    """
    return max(1, int((file_size_mb / max_batch_size_mb) + 0.5))

def calculate_characters_and_tokens(texts):
    """Calculate total characters and estimate tokens in the texts.
    
    Args:
        texts: List of texts
        
    Returns:
        Tuple of (total characters, estimated tokens)
    """
    total_chars = sum(len(text) for text in texts)
    # Rough estimate: 1 token is roughly 4 characters on average for English text
    estimated_tokens = total_chars / 4
    return total_chars, estimated_tokens

def main():
    parser = argparse.ArgumentParser(description="Check the size of JSONL file for batch processing")
    parser.add_argument("--input", "-i", default="source-dataset/cleaned_aae_dataset.csv", 
                        help="Path to input CSV file with tweets")
    parser.add_argument("--output-jsonl", "-o", help="Path to output JSONL file (optional)")
    parser.add_argument("--samples", "-s", type=int, default=None, 
                        help="Number of samples to process (default: all)")
    parser.add_argument("--batch-type", "-t", choices=["sentiment", "translation"], 
                        default="sentiment", help="Type of batch to create")
    args = parser.parse_args()
    
    # Load the dataset
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded dataset with {len(df)} records")
        
        if args.samples:
            df = df.sample(args.samples, random_state=42) if len(df) > args.samples else df
            logger.info(f"Sampled {len(df)} records")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        sys.exit(1)
    
    # Extract texts
    if "text" in df.columns:
        texts = df["text"].tolist()
    else:
        logger.error("Input CSV must have 'text' column")
        sys.exit(1)
    
    # Calculate text statistics
    total_chars, estimated_tokens = calculate_characters_and_tokens(texts)
    
    # Create batch file and measure size
    try:
        if args.batch_type == "sentiment":
            file_path, file_size_mb = create_batch_request_file_for_sentiment(texts, output_path=args.output_jsonl)
        else:
            file_path, file_size_mb = create_batch_request_file_for_translation(texts, output_path=args.output_jsonl)
        
        # Calculate batches needed
        batches_needed = estimate_batches_needed(file_size_mb)
        
        # Print results
        logger.info(f"Dataset summary:")
        logger.info(f"  - Total records: {len(texts)}")
        logger.info(f"  - Total characters: {total_chars:,}")
        logger.info(f"  - Estimated tokens: {int(estimated_tokens):,}")
        logger.info(f"JSONL file details:")
        logger.info(f"  - File path: {file_path}")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        logger.info(f"  - Batches needed (100MB limit): {batches_needed}")
        
        recommended_batch_size = max(1, len(texts) // batches_needed)
        logger.info(f"Recommended batch size: {recommended_batch_size} records")
        
        if not args.output_jsonl:
            # Clean up temporary file
            os.unlink(file_path)
            logger.info("Temporary file deleted")
        
    except Exception as e:
        logger.error(f"Error in batch file size check: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()