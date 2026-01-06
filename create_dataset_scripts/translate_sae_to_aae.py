"""
Script to translate SAE tweets back to AAE using GPT-4.1 Batch API with three modes:
1. submit: Submit batch and exit
2. poll: Check status of existing batch and save results when complete
3. retrieve: Retrieve and save results from a completed batch

Includes SMS notifications using Twilio for important job status updates.
"""

import os
import sys
import json
import pandas as pd
import logging
import argparse
import asyncio
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GPT41BatchInterface
from utils.notifications import (
    notify_batch_submitted, 
    notify_batch_status, 
    notify_results_saved, 
    notify_error
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sae_to_aae_translation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Task name for notifications
TASK_NAME = "SAE to AAE translation"

def load_sae_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the SAE dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        num_samples: Number of samples to load, if None, load all
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(df)} records")
        
        if num_samples:
            df = df.sample(num_samples, random_state=42) if len(df) > num_samples else df
            logger.info(f"Sampled {len(df)} records")
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        notify_error(TASK_NAME, f"Failed to load dataset: {str(e)}")
        raise

def submit_translation_batch(texts: List[str], api_key: str) -> str:
    """
    Submit a batch translation job for SAE texts to AAE.
    
    Args:
        texts: List of SAE texts to translate
        api_key: OpenAI API key
        
    Returns:
        Batch ID for the submitted job
    """
    model = GPT41BatchInterface(api_key)
    
    # Define the translation prompt template
    translation_prompt_template = """
    Translate the following tweet from Standard American English (SAE) to African American English (AAE).
    Preserve the meaning, tone, and intent of the original tweet.
    Only change dialectical features while maintaining the original message.
    Original tweet (SAE): {text}
    African American English translation:
    """
    
    # Create batch file for the entire dataset
    logger.info(f"Creating batch file for {len(texts)} texts")
    batch_file_path, file_size_mb = model.create_batch_request_file(texts, translation_prompt_template)
    logger.info(f"Batch file created, size: {file_size_mb:.2f} MB")
    
    # Submit batch
    try:
        batch_id = model.submit_batch(batch_file_path, "SAE to AAE translation batch")
        logger.info(f"Batch job submitted with ID: {batch_id}")
        
        # Send notification
        notify_batch_submitted(batch_id, TASK_NAME, len(texts))
        
        # Save batch information to file for later retrieval
        batch_info = {
            'batch_id': batch_id,
            'num_texts': len(texts),
            'file_size_mb': file_size_mb,
            'status': 'submitted'
        }
        
        with open('sae_to_aae_batch_info.json', 'w') as f:
            json.dump(batch_info, f)
            
        logger.info(f"Batch information saved to sae_to_aae_batch_info.json")
        
        # Clean up batch file
        try:
            os.unlink(batch_file_path)
        except Exception as e:
            logger.warning(f"Error deleting batch file: {str(e)}")
        
        return batch_id
    except Exception as e:
        logger.error(f"Error submitting batch: {str(e)}")
        notify_error(TASK_NAME, f"Failed to submit batch: {str(e)}")
        raise

def poll_translation_batch(batch_id: str, api_key: str, df: pd.DataFrame, output_path: str) -> bool:
    """
    Poll for batch status and save results if complete.
    
    Args:
        batch_id: ID of the batch to poll
        api_key: OpenAI API key
        df: Original dataframe with texts
        output_path: Path to save results
        
    Returns:
        True if batch completed and results saved, False otherwise
    """
    model = GPT41BatchInterface(api_key)
    
    # Check status
    status = model.check_batch_status(batch_id)
    logger.info(f"Batch {batch_id} status: {status['status']}")
    
    # Send status notification
    notify_batch_status(
        batch_id, 
        TASK_NAME, 
        status['status'], 
        status['progress']
    )
    
    # Update batch info file
    try:
        with open('sae_to_aae_batch_info.json', 'r') as f:
            batch_info = json.load(f)
        
        batch_info['status'] = status['status']
        
        with open('sae_to_aae_batch_info.json', 'w') as f:
            json.dump(batch_info, f)
    except Exception as e:
        logger.warning(f"Error updating batch info file: {str(e)}")
    
    # If completed, retrieve results and save
    if status['status'] == 'completed':
        logger.info(f"Batch {batch_id} is complete. Retrieving results...")
        
        # Get results
        try:
            batch_results = model.get_batch_results(batch_id)
            responses = model.extract_responses_from_batch(batch_results)
            
            # Process responses
            sae_texts = df['sae_text'].tolist() if 'sae_text' in df.columns else []
            original_aae_texts = df['original_text'].tolist() if 'original_text' in df.columns else [""] * len(sae_texts)
            
            all_translations = []
            
            for i in range(len(sae_texts)):
                custom_id = f"request-{i}"
                response = responses.get(custom_id, f"ERROR: No response for {custom_id}")
                
                # Extract translation
                if "ERROR:" in response:
                    all_translations.append(f"TRANSLATION_FAILED: {sae_texts[i]}")
                else:
                    # Extract from formatted response if needed
                    if "African American English translation:" in response:
                        parts = response.split("African American English translation:")
                        if len(parts) > 1:
                            all_translations.append(parts[1].strip())
                        else:
                            all_translations.append(f"TRANSLATION_FAILED: {sae_texts[i]}")
                    else:
                        # Use whole response
                        all_translations.append(response.strip())
            
            # Create output dataframe
            output_df = pd.DataFrame({
                "original_aae_text": original_aae_texts,
                "sae_text": sae_texts,
                "aae_from_sae_text": all_translations
            })
            
            # Save output
            output_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(output_df)} translations to {output_path}")
            
            # Send notification
            notify_results_saved(TASK_NAME, len(output_df), output_path)
            
            # Update batch info
            batch_info['results_saved'] = True
            with open('sae_to_aae_batch_info.json', 'w') as f:
                json.dump(batch_info, f)
                
            return True
        except Exception as e:
            logger.error(f"Error retrieving batch results: {str(e)}")
            notify_error(TASK_NAME, f"Failed to retrieve batch results: {str(e)}")
            return False
    else:
        # Not completed yet
        if status['progress']['total'] > 0:
            percent_complete = (status['progress']['completed'] + status['progress']['failed']) / status['progress']['total'] * 100
            logger.info(f"Progress: {percent_complete:.1f}% ({status['progress']['completed']}/{status['progress']['total']} completed)")
        return False

def retrieve_batch_results(batch_id: str, api_key: str, df: pd.DataFrame, output_path: str) -> bool:
    """
    Retrieve results from a completed batch and save them.
    
    Args:
        batch_id: ID of the completed batch
        api_key: OpenAI API key
        df: Original dataframe with texts
        output_path: Path to save results
        
    Returns:
        True if results successfully saved, False otherwise
    """
    model = GPT41BatchInterface(api_key)
    
    # Check if batch is completed
    status = model.check_batch_status(batch_id)
    if status['status'] != 'completed':
        error_msg = f"Batch {batch_id} is not completed (status: {status['status']}). Cannot retrieve results."
        logger.error(error_msg)
        notify_error(TASK_NAME, error_msg)
        return False
    
    # Retrieve and save results
    try:
        logger.info(f"Retrieving results for batch {batch_id}...")
        batch_results = model.get_batch_results(batch_id)
        responses = model.extract_responses_from_batch(batch_results)
        
        # Process responses
        sae_texts = df['sae_text'].tolist() if 'sae_text' in df.columns else []
        original_aae_texts = df['original_text'].tolist() if 'original_text' in df.columns else [""] * len(sae_texts)
        
        all_translations = []
        
        for i in range(len(sae_texts)):
            custom_id = f"request-{i}"
            response = responses.get(custom_id, f"ERROR: No response for {custom_id}")
            
            # Extract translation
            if "ERROR:" in response:
                all_translations.append(f"TRANSLATION_FAILED: {sae_texts[i]}")
            else:
                # Extract from formatted response if needed
                if "African American English translation:" in response:
                    parts = response.split("African American English translation:")
                    if len(parts) > 1:
                        all_translations.append(parts[1].strip())
                    else:
                        all_translations.append(f"TRANSLATION_FAILED: {sae_texts[i]}")
                else:
                    # Use whole response
                    all_translations.append(response.strip())
        
        # Create output dataframe
        output_df = pd.DataFrame({
            "original_aae_text": original_aae_texts,
            "sae_text": sae_texts,
            "aae_from_sae_text": all_translations
        })
        
        # Save output
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(output_df)} translations to {output_path}")
        
        # Send notification
        notify_results_saved(TASK_NAME, len(output_df), output_path)
        
        # Update batch info
        try:
            with open('sae_to_aae_batch_info.json', 'r') as f:
                batch_info = json.load(f)
            
            batch_info['status'] = status['status']
            batch_info['results_saved'] = True
            
            with open('sae_to_aae_batch_info.json', 'w') as f:
                json.dump(batch_info, f)
        except Exception as e:
            logger.warning(f"Error updating batch info file: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error retrieving batch results: {str(e)}")
        notify_error(TASK_NAME, f"Failed to retrieve batch results: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Translate SAE tweets back to AAE using GPT-4.1 Batch API")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with SAE texts")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file for AAE translations")
    parser.add_argument("--mode", "-m", choices=["submit", "poll", "retrieve"], required=True, 
                        help="Operation mode: submit a new batch, poll existing batch, or retrieve completed results")
    parser.add_argument("--batch-id", "-b", help="Batch ID for poll or retrieve modes")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--api-key", "-k", help="OpenAI API key (if not provided, will use environment variable)")
    parser.add_argument("--enable-notifications", "-n", action="store_true", 
                        help="Enable SMS notifications (requires Twilio credentials in environment)")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Set global flag for notifications (overrides the imported functions if disabled)
    if not args.enable_notifications:
        global notify_batch_submitted, notify_batch_status, notify_results_saved, notify_error
        notify_batch_submitted = lambda *args, **kwargs: False
        notify_batch_status = lambda *args, **kwargs: False
        notify_results_saved = lambda *args, **kwargs: False
        notify_error = lambda *args, **kwargs: False
        logger.info("SMS notifications disabled")
    else:
        logger.info("SMS notifications enabled")
    
    # Load the dataset
    try:
        df = load_sae_dataset(args.input, args.samples)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Get texts for translation
    if 'sae_text' not in df.columns:
        error_msg = "Input CSV must have 'sae_text' column with SAE translations"
        logger.error(error_msg)
        notify_error(TASK_NAME, error_msg)
        sys.exit(1)
    
    sae_texts = df['sae_text'].tolist()
    
    # Execute based on mode
    if args.mode == "submit":
        try:
            batch_id = submit_translation_batch(sae_texts, api_key)
            logger.info(f"Successfully submitted batch job with ID: {batch_id}")
            logger.info(f"To check status later, run:")
            logger.info(f"python {sys.argv[0]} --mode poll --batch-id {batch_id} --input {args.input} --output {args.output}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit batch job: {str(e)}")
            notify_error(TASK_NAME, f"Failed to submit batch job: {str(e)}")
            return False
            
    elif args.mode == "poll":
        if not args.batch_id:
            # Try to read from batch info file
            try:
                with open('sae_to_aae_batch_info.json', 'r') as f:
                    batch_info = json.load(f)
                batch_id = batch_info['batch_id']
                logger.info(f"Using batch ID from batch info file: {batch_id}")
            except:
                error_msg = "No batch ID provided. Use --batch-id or ensure sae_to_aae_batch_info.json exists."
                logger.error(error_msg)
                notify_error(TASK_NAME, error_msg)
                sys.exit(1)
        else:
            batch_id = args.batch_id
            
        result = poll_translation_batch(batch_id, api_key, df, args.output)
        if result:
            logger.info("Batch completed and results saved.")
        else:
            logger.info(f"Batch is still processing. Run this command again later to check status.")
        return result
            
    elif args.mode == "retrieve":
        if not args.batch_id:
            # Try to read from batch info file
            try:
                with open('sae_to_aae_batch_info.json', 'r') as f:
                    batch_info = json.load(f)
                batch_id = batch_info['batch_id']
                logger.info(f"Using batch ID from batch info file: {batch_id}")
            except:
                error_msg = "No batch ID provided. Use --batch-id or ensure sae_to_aae_batch_info.json exists."
                logger.error(error_msg)
                notify_error(TASK_NAME, error_msg)
                sys.exit(1)
        else:
            batch_id = args.batch_id
            
        result = retrieve_batch_results(batch_id, api_key, df, args.output)
        if result:
            logger.info("Results successfully retrieved and saved.")
        else:
            logger.error("Failed to retrieve batch results.")
        return result
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)