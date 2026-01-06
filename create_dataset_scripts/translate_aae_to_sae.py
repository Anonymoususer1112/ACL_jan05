"""
Script to translate AAE tweets to SAE using GPT-4.1 Batch API with three modes:
1. submit: Submit batch and exit
2. poll: Check status of existing batch and save results when complete
3. retrieve: Retrieve and save results from a completed batch

Includes SMS notifications using Twilio for important job status updates.
"""

import os
import sys
import json
import time
import signal
import pandas as pd
import logging
import argparse
import asyncio
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GPT41BatchInterface
from utils.notifications import (
    notify_batch_submitted, 
    notify_batch_status, 
    notify_results_saved, 
    notify_error,
    _is_twilio_configured
)

# Import Hugging Face utilities for dataset upload
try:
    from datasets import Dataset
    from huggingface_hub import HfApi, HfFolder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aae_to_sae_translation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Task name for notifications
TASK_NAME = "AAE to SAE translation"

# Notification frequency options
class NotificationFrequency(Enum):
    EVERY_MINUTE = "every minute"
    EVERY_10_MINS = "every 10 mins" 
    EVERY_QUARTER_PROGRESS = "every quarter of progress"
    EVERY_90_MINS = "every 1.5 hours"

# Flag to control continuous polling
POLLING_ACTIVE = True

def load_aae_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the AAE dataset from a CSV file.
    Opens the file as text, extracts headers from first line, then treats each line as a whole.
    
    Args:
        filepath: Path to the CSV file
        num_samples: Number of samples to load, if None, load all
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Open as text file
        with open(filepath, 'r', encoding='utf-8') as f:
            # Get headers from first line
            header_line = f.readline().strip()
            headers = [h.strip() for h in header_line.split(',')]
            
            # Read remaining lines without splitting on commas
            lines = [line.strip() for line in f if line.strip()]
        
        # Sample if requested
        if num_samples and num_samples < len(lines):
            import random
            random.seed(42)
            lines = random.sample(lines, num_samples)
        
        # Create DataFrame with text column (using first header as column name)
        column_name = 'text'  # Use standard name for compatibility with rest of code
        
        df = pd.DataFrame({
            column_name: lines
        })
        
        logger.info(f"Loaded dataset as text: {len(df)} records")
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        notify_error(TASK_NAME, f"Failed to load dataset: {str(e)}")
        raise

def submit_translation_batch(texts: List[str], api_key: str) -> str:
    """
    Submit a batch translation job for AAE texts to SAE.
    
    Args:
        texts: List of AAE texts to translate
        api_key: OpenAI API key
        
    Returns:
        Batch ID for the submitted job
    """
    model = GPT41BatchInterface(api_key)
    
    # Define the translation prompt template
    translation_prompt_template = """
    Translate the following tweet from African American English (AAE) to Standard American English (SAE).
    Preserve the meaning, tone, and intent of the original tweet.
    Only change dialectical features while maintaining the original message.
    Original tweet (AAE): {text}
    Standard American English translation:
    """
    
    # Create batch file for the entire dataset
    logger.info(f"Creating batch file for {len(texts)} texts")
    batch_file_path, file_size_mb = model.create_batch_request_file(texts, translation_prompt_template)
    logger.info(f"Batch file created, size: {file_size_mb:.2f} MB")
    
    # Submit batch
    try:
        batch_id = model.submit_batch(batch_file_path, "AAE to SAE translation batch")
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
        
        with open('aae_to_sae_batch_info.json', 'w') as f:
            json.dump(batch_info, f)
            
        logger.info(f"Batch information saved to aae_to_sae_batch_info.json")
        
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
    
    # Always send status notification
    notify_status = notify_batch_status(
        batch_id, 
        TASK_NAME, 
        status['status'], 
        status['progress']
    )
    logger.info(f"Status notification sent: {notify_status}")
    
    # Update batch info file
    try:
        with open('aae_to_sae_batch_info.json', 'r') as f:
            batch_info = json.load(f)
        
        batch_info['status'] = status['status']
        if status['progress']['total'] > 0:
            percent_complete = (status['progress']['completed'] + status['progress']['failed']) / status['progress']['total'] * 100
            batch_info['progress_percent'] = percent_complete
            logger.info(f"Progress: {percent_complete:.1f}% ({status['progress']['completed']}/{status['progress']['total']} completed)")
        
        with open('aae_to_sae_batch_info.json', 'w') as f:
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
            aae_texts = df['text'].tolist()
            
            all_translations = []
            
            for i in range(len(aae_texts)):
                custom_id = f"request-{i}"
                response = responses.get(custom_id, f"ERROR: No response for {custom_id}")
                
                # Extract translation
                if "ERROR:" in response:
                    all_translations.append(f"TRANSLATION_FAILED: {aae_texts[i]}")
                else:
                    # Extract from formatted response if needed
                    if "Standard American English translation:" in response:
                        parts = response.split("Standard American English translation:")
                        if len(parts) > 1:
                            all_translations.append(parts[1].strip())
                        else:
                            all_translations.append(f"TRANSLATION_FAILED: {aae_texts[i]}")
                    else:
                        # Use whole response
                        all_translations.append(response.strip())
            
            # Create output dataframe
            output_df = pd.DataFrame({
                "original_text": aae_texts,
                "sae_text": all_translations
            })
            
            # Save output
            output_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(output_df)} translations to {output_path}")
            
            # Send notification
            notify_saved = notify_results_saved(TASK_NAME, len(output_df), output_path)
            logger.info(f"Results saved notification sent: {notify_saved}")
            
            # Update batch info
            batch_info['results_saved'] = True
            with open('aae_to_sae_batch_info.json', 'w') as f:
                json.dump(batch_info, f)
                
            return True
        except Exception as e:
            logger.error(f"Error retrieving batch results: {str(e)}")
            notify_error(TASK_NAME, f"Failed to retrieve batch results: {str(e)}")
            return False
    else:
        # Not completed yet
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
        aae_texts = df['text'].tolist()
        
        all_translations = []
        
        for i in range(len(aae_texts)):
            custom_id = f"request-{i}"
            response = responses.get(custom_id, f"ERROR: No response for {custom_id}")
            
            # Extract translation
            if "ERROR:" in response:
                all_translations.append(f"TRANSLATION_FAILED: {aae_texts[i]}")
            else:
                # Extract from formatted response if needed
                if "Standard American English translation:" in response:
                    parts = response.split("Standard American English translation:")
                    if len(parts) > 1:
                        all_translations.append(parts[1].strip())
                    else:
                        all_translations.append(f"TRANSLATION_FAILED: {aae_texts[i]}")
                else:
                    # Use whole response
                    all_translations.append(response.strip())
        
        # Create output dataframe
        output_df = pd.DataFrame({
            "original_text": aae_texts,
            "sae_text": all_translations
        })
        
        # Save output
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(output_df)} translations to {output_path}")
        
        # Try uploading to Hugging Face
        hf_upload_success, hf_result = upload_dataset_to_huggingface(output_path)
        dataset_url = None
        
        if hf_upload_success:
            dataset_url = hf_result
            logger.info(f"Dataset uploaded to Hugging Face: {dataset_url}")
        else:
            logger.warning(f"Failed to upload to Hugging Face: {hf_result}")
        
        # Send notification with dataset URL if available
        notify_results_saved(TASK_NAME, len(output_df), output_path, dataset_url)
        
        # Update batch info
        try:
            with open('aae_to_sae_batch_info.json', 'r') as f:
                batch_info = json.load(f)
            
            batch_info['status'] = status['status']
            batch_info['results_saved'] = True
            if dataset_url:
                batch_info['huggingface_url'] = dataset_url
            
            with open('aae_to_sae_batch_info.json', 'w') as f:
                json.dump(batch_info, f)
        except Exception as e:
            logger.warning(f"Error updating batch info file: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error retrieving batch results: {str(e)}")
        notify_error(TASK_NAME, f"Failed to retrieve batch results: {str(e)}")
        return False

def signal_handler(sig, frame):
    """Handle interrupt signal to gracefully stop polling."""
    global POLLING_ACTIVE
    logger.info("Received interrupt signal. Stopping polling after current check completes...")
    POLLING_ACTIVE = False

def continuous_poll_translation_batch(batch_id: str, api_key: str, df: pd.DataFrame, output_path: str, 
                                      notification_frequency: NotificationFrequency) -> bool:
    """
    Continuously poll a batch translation job until complete or interrupted.
    
    Args:
        batch_id: ID of the batch to poll
        api_key: OpenAI API key
        df: Original dataframe with texts
        output_path: Path to save results
        notification_frequency: How often to send notifications
        
    Returns:
        True if batch completed successfully, False otherwise
    """
    global POLLING_ACTIVE
    POLLING_ACTIVE = True
    
    # Set up signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting continuous polling for batch {batch_id}")
    logger.info(f"Notification frequency: {notification_frequency.value}")
    logger.info("Press Ctrl+C to stop polling")
    
    model = GPT41BatchInterface(api_key)
    
    # For tracking progress-based notifications
    last_notification_time = 0
    last_progress_quarter = -1  # -1, 0, 1, 2, 3 for 0%, 25%, 50%, 75%, 100%
    completed = False
    
    while POLLING_ACTIVE:
        try:
            # Check status
            status = model.check_batch_status(batch_id)
            current_time = time.time()
            
            # Calculate progress percentage if available
            progress_percent = 0
            if status['progress']['total'] > 0:
                progress_percent = (status['progress']['completed'] + status['progress']['failed']) / status['progress']['total'] * 100
                current_quarter = int(progress_percent / 25)
            else:
                current_quarter = -1
            
            # Log current status
            logger.info(f"Batch {batch_id} status: {status['status']}")
            if status['progress']['total'] > 0:
                logger.info(f"Progress: {progress_percent:.1f}% ({status['progress']['completed']}/{status['progress']['total']} completed)")
            
            # Determine if we should send a notification based on frequency setting
            send_notification = False
            
            if notification_frequency == NotificationFrequency.EVERY_MINUTE:
                # Check if at least a minute has passed since last notification
                if current_time - last_notification_time >= 60:
                    send_notification = True
                    
            elif notification_frequency == NotificationFrequency.EVERY_10_MINS:
                # Check if at least 10 minutes have passed since last notification
                if current_time - last_notification_time >= 600:
                    send_notification = True
                    
            elif notification_frequency == NotificationFrequency.EVERY_QUARTER_PROGRESS:
                # Check if we've reached a new quarter of progress (25%, 50%, 75%, 100%)
                if current_quarter > last_progress_quarter:
                    send_notification = True
                    last_progress_quarter = current_quarter

            # Also check if we should notify every 1.5 hours
            elif notification_frequency == NotificationFrequency.EVERY_90_MINS:
                # Check if at least 90 minutes (1.5 hours) have passed since last notification
                if current_time - last_notification_time >= 5400:  # 90 * 60 = 5400 seconds
                    send_notification = True
            
            # Send notification if needed
            if send_notification:
                notify_batch_status(
                    batch_id, 
                    TASK_NAME, 
                    status['status'], 
                    status['progress']
                )
                last_notification_time = current_time
                
            # Update batch info file
            try:
                with open('aae_to_sae_batch_info.json', 'r') as f:
                    batch_info = json.load(f)
                
                batch_info['status'] = status['status']
                if status['progress']['total'] > 0:
                    batch_info['progress_percent'] = progress_percent
                
                with open('aae_to_sae_batch_info.json', 'w') as f:
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
                    aae_texts = df['text'].tolist()
                    all_translations = []
                    
                    for i in range(len(aae_texts)):
                        custom_id = f"request-{i}"
                        response = responses.get(custom_id, f"ERROR: No response for {custom_id}")
                        
                        # Extract translation
                        if "ERROR:" in response:
                            all_translations.append(f"TRANSLATION_FAILED: {aae_texts[i]}")
                        else:
                            # Extract from formatted response if needed
                            if "Standard American English translation:" in response:
                                parts = response.split("Standard American English translation:")
                                if len(parts) > 1:
                                    all_translations.append(parts[1].strip())
                                else:
                                    all_translations.append(f"TRANSLATION_FAILED: {aae_texts[i]}")
                            else:
                                # Use whole response
                                all_translations.append(response.strip())
                    
                    # Create output dataframe
                    output_df = pd.DataFrame({
                        "original_text": aae_texts,
                        "sae_text": all_translations
                    })
                    
                    # Save output
                    output_df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(output_df)} translations to {output_path}")
                    
                    # Try uploading to Hugging Face
                    hf_upload_success, hf_result = upload_dataset_to_huggingface(output_path)
                    dataset_url = None
                    
                    if hf_upload_success:
                        dataset_url = hf_result
                        logger.info(f"Dataset uploaded to Hugging Face: {dataset_url}")
                        # Save the URL in batch info
                        batch_info['huggingface_url'] = dataset_url
                    else:
                        logger.warning(f"Failed to upload to Hugging Face: {hf_result}")
                    
                    # Send notification with dataset URL if available
                    notify_results_saved(TASK_NAME, len(output_df), output_path, dataset_url)
                    
                    # Update batch info
                    batch_info['results_saved'] = True
                    with open('aae_to_sae_batch_info.json', 'w') as f:
                        json.dump(batch_info, f)
                    
                    completed = True
                    break
                    
                except Exception as e:
                    logger.error(f"Error retrieving batch results: {str(e)}")
                    notify_error(TASK_NAME, f"Failed to retrieve batch results: {str(e)}")
                    return False
            
            # If failed, cancelled, or expired, stop polling
            elif status['status'] in ['failed', 'cancelled', 'expired']:
                logger.error(f"Batch job ended with status: {status['status']}")
                notify_error(TASK_NAME, f"Batch job ended with status: {status['status']}")
                break
                
            # Wait before next poll - adjust polling interval based on progress
            # More frequent at the beginning, less frequent when close to completion
            if progress_percent < 25:
                sleep_time = 30  # Poll every 30 seconds at the beginning
            elif progress_percent < 75:
                sleep_time = 60  # Poll every minute in the middle
            else:
                sleep_time = 30  # Poll more frequently near completion
                
            logger.info(f"Next status check in {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error during polling: {str(e)}")
            notify_error(TASK_NAME, f"Error during polling: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying after error
    
    if not POLLING_ACTIVE:
        logger.info("Polling stopped by user")
        
    return completed

def upload_dataset_to_huggingface(csv_path: str, dataset_name: str = None) -> Tuple[bool, str]:
    """
    Upload a dataset to Hugging Face Hub and return the URL.
    
    Args:
        csv_path: Path to the CSV file to upload
        dataset_name: Name for the dataset on Hugging Face (default: auto-generated based on CSV name)
        
    Returns:
        Tuple of (success, url_or_error_message)
    """
    if not HF_AVAILABLE:
        logger.warning("Hugging Face libraries not available. Cannot upload dataset.")
        return False, "Hugging Face libraries not installed"
    
    # Check if HF_TOKEN is set in environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set in environment. Cannot upload dataset.")
        return False, "HF_TOKEN not configured in environment"
        
    try:
        # Auto-generate dataset name if not provided
        if not dataset_name:
            # Create a unique name based on the file name and date
            base_name = os.path.basename(csv_path).split('.')[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"aae-sae-translation-{base_name}-{timestamp}"
            
            # Add username prefix if not already present
            # Get username from huggingface-cli if available
            username = "furquan"
            if username and not dataset_name.startswith(f"{username}/"):
                dataset_name = f"{username}/{dataset_name}"
                
        logger.info(f"Uploading dataset to Hugging Face as: {dataset_name}")
        
        # Load the CSV as a Hugging Face dataset
        df = pd.read_csv(csv_path)
        hf_dataset = Dataset.from_pandas(df)
        
        # Push the dataset to the hub
        hf_dataset.push_to_hub(dataset_name)
        
        # Construct the URL for the dataset
        dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
        logger.info(f"Dataset successfully uploaded to: {dataset_url}")
        
        return True, dataset_url
        
    except Exception as e:
        error_msg = f"Failed to upload dataset to Hugging Face: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="Translate AAE tweets to SAE using GPT-4.1 Batch API")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with AAE texts")
    parser.add_argument("--output", "-o",  required=True, help="Path to output CSV file for SAE translations")
    parser.add_argument("--mode", "-m", choices=["submit", "poll", "retrieve", "continuous-poll"], required=True, 
                        help="Operation mode: submit a new batch, poll existing batch, continuously poll batch, or retrieve completed results")
    parser.add_argument("--batch_id", "-b", help="Batch ID for poll or retrieve modes")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--api-key", "-k", help="OpenAI API key (if not provided, will use environment variable)")
    parser.add_argument("--enable-notifications", "-n", action="store_true", 
                        help="Enable SMS notifications (requires Twilio credentials in environment)")
    parser.add_argument("--notification-frequency", "-f", choices=["every minute", "every 10 mins", "every quarter of progress", "every 1.5 hours"],
                        default="every quarter of progress",
                        help="How often to send status notifications when using continuous-poll mode")
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
        logger.info("SMS notifications disabled. Use --enable-notifications flag to receive SMS updates.")
        # Check if Twilio credentials are properly configured
        if not _is_twilio_configured():
            logger.warning("Note: Twilio credentials are not properly configured in environment variables.")
            logger.warning("Even if you use --enable-notifications, SMS won't be sent until Twilio is configured.")
    else:
        # Check if Twilio is properly configured
        if not _is_twilio_configured():
            logger.warning("Twilio credentials are not properly configured. SMS notifications won't be sent.")
            logger.warning("Please check your environment variables for Twilio configuration.")
        else:
            logger.info("SMS notifications enabled. You will receive updates via SMS.")
    
    # Load the dataset
    try:
        df = load_aae_dataset(args.input, args.samples)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Get texts for translation
    if 'text' not in df.columns:
        error_msg = "Input CSV must have 'text' column with AAE texts"
        logger.error(error_msg)
        notify_error(TASK_NAME, error_msg)
        sys.exit(1)
    
    aae_texts = df['text'].tolist()
    
    # Parse notification frequency
    notification_freq = None
    if args.notification_frequency == "every minute":
        notification_freq = NotificationFrequency.EVERY_MINUTE
    elif args.notification_frequency == "every 10 mins":
        notification_freq = NotificationFrequency.EVERY_10_MINS
    else:  # Default to every quarter of progress
        notification_freq = NotificationFrequency.EVERY_QUARTER_PROGRESS
    
    # Execute based on mode
    if args.mode == "submit":
        try:
            batch_id = submit_translation_batch(aae_texts, api_key)
            logger.info(f"Successfully submitted batch job with ID: {batch_id}")
            logger.info(f"To check status later, run:")
            logger.info(f"python {sys.argv[0]} --mode poll --batch-id {batch_id} --input {args.input} --output {args.output}")
            logger.info(f"Or for continuous polling until completion:")
            logger.info(f"python {sys.argv[0]} --mode continuous-poll --batch-id {batch_id} --input {args.input} --output {args.output}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit batch job: {str(e)}")
            notify_error(TASK_NAME, f"Failed to submit batch job: {str(e)}")
            return False
            
    elif args.mode == "poll":
        if not args.batch_id:
            # Try to read from batch info file
            try:
                with open('aae_to_sae_batch_info.json', 'r') as f:
                    batch_info = json.load(f)
                batch_id = batch_info['batch_id']
                logger.info(f"Using batch ID from batch info file: {batch_id}")
            except:
                error_msg = "No batch ID provided. Use --batch-id or ensure aae_to_sae_batch_info.json exists."
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
        
    elif args.mode == "continuous-poll":
        if not args.batch_id:
            # Try to read from batch info file
            try:
                with open('aae_to_sae_batch_info.json', 'r') as f:
                    batch_info = json.load(f)
                batch_id = batch_info['batch_id']
                logger.info(f"Using batch ID from batch info file: {batch_id}")
            except:
                error_msg = "No batch ID provided. Use --batch-id or ensure aae_to_sae_batch_info.json exists."
                logger.error(error_msg)
                notify_error(TASK_NAME, error_msg)
                sys.exit(1)
        else:
            batch_id = args.batch_id
        
        logger.info(f"Starting continuous polling with notification frequency: {args.notification_frequency}")
        result = continuous_poll_translation_batch(batch_id, api_key, df, args.output, notification_freq)
        
        if result:
            logger.info("Batch completed successfully and results saved.")
        else:
            logger.info("Continuous polling ended without completion. You can resume polling later.")
        return result
            
    elif args.mode == "retrieve":
        if not args.batch_id:
            # Try to read from batch info file
            try:
                with open('aae_to_sae_batch_info.json', 'r') as f:
                    batch_info = json.load(f)
                batch_id = batch_info['batch_id']
                logger.info(f"Using batch ID from batch info file: {batch_id}")
            except:
                error_msg = "No batch ID provided. Use --batch-id or ensure aae_to_sae_batch_info.json exists."
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