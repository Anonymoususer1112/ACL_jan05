from ast import arg, parse
import os

import random
import sys
import csv
from banal import chunked_iter
from cv2 import log
import pandas as pd
import logging
import argparse
import asyncio
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from zmq import has

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GPT41BatchInterface, ClaudeHaikuInterface, Phi4VllmInterface, ClaudeBatchInterface

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("preference_multiple_choice.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the preference multiple choice dataset from a CSV file.
    
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
        raise

def get_model_interface(model_name: str, api_key: Optional[str] = None):
    """
    Get the appropriate model interface based on the model name.
    
    Args:
        model_name: Name of the model to use
        api_key: API key for the model (if applicable)
        
    Returns:
        Model interface object
    """
    if model_name == 'gpt4o_mini':
        from models import GPT4oMiniInterface
        return GPT4oMiniInterface(api_key=api_key)
    elif model_name == 'gpt41_batch':
        from models import GPT41BatchInterface
        return GPT41BatchInterface(api_key=api_key)
    elif model_name == 'claude_haiku':
        from models import ClaudeHaikuInterface
        return ClaudeHaikuInterface(api_key=api_key)
    elif model_name == 'phi3_medium':
        from models import Phi3MediumInterface
        return Phi3MediumInterface()
    elif model_name == 'phi4_vllm':
        from models import Phi4VllmInterface
        return Phi4VllmInterface()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
def main():
    parser = argparse.ArgumentParser(description="Preference Multiple Choice Task Evaluation")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model", "-m", required=True, 
                        choices=['gpt4o_mini', 'gpt41_batch', 'claude_haiku', 'phi3_medium', 'phi4_vllm'], 
                        help="Model to use for evaluation")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--api-key", "-k", help="API key for the selected model (if applicable)")
    parser.add_argument("--sae-context-column", default="sae_context", help="Column name containing SAE context")
    parser.add_argument("--aae-context-column", default="aae_context", help="Column name containing AAE context")
    parser.add_argument("--sae-cont-column", default="sae_continuation", help="Column name containing SAE continuation")
    parser.add_argument("--aae-cont-column", default="aae_continuation", help="Column name containing AAE continuation")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.input, num_samples=args.samples)

    # Validate columns 
    for col in [args.sae_context_column, args.aae_context_column,
                args.sae_cont_column, args.aae_cont_column]:
        if col not in df.columns:
            logger.error(f"Specified column '{col}' not found in dataset")
            sys.exit(1)

    # Get model interface
    try:
        model = get_model_interface(args.model, api_key=args.api_key)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    df_indexed = df.reset_index() 
    rows = df_indexed.to_dict(orient='records')

    total = len(rows)
    logger.info(f"Loaded {total} records for evaluation")

    def calculate_log_metrics(row, context_variant: str, model):
        """
        Calculate log probabilities and log differences for SAE and AAE continuations.
        
        Args:
            row: A single record from the dataset
            context_variant: 'sae' or 'aae' indicating which context to use
            model: Model interface object
        Returns:
            Dictionary with log probabilities and log differences
        """


        log_prob_aae, log_prob_sae, log_difference = model.calculate_log_difference_for_preference(
            context_sae=row[args.sae_context_column],
            continuation_sae=row[args.sae_cont_column],
            context_aae=row[args.aae_context_column],
            continuation_aae=row[args.aae_cont_column],
            context_variant=context_variant 
        )

        # Return a dictionary of results for easy merge
        return {
            f'log_prob_aae_{context_variant}': log_prob_aae,
            f'log_prob_sae_{context_variant}': log_prob_sae,
            f'log_difference_{context_variant}': log_difference
        }
    
    if not hasattr(model, 'calculate_log_difference_for_preference'):
        logger.error("Selected model does not support log difference calculation for preference task.")
        sys.exit(1)

    all_metrics: List[Dict[str, Any]] = []
    logger.info("Starting log probability scoring for {total} records using {args.model}")


    # Iterate over the dataset in batches
    for batch in tqdm(chunked_iter(list(enumerate(rows)), args.batch_size),
                    total=(total + args.batch_size - 1) // args.batch_size):
        
        for idx, row in batch:

            # 1) SAE context
            metrics_sae = calculate_log_metrics(row, context_variant="SAE", model=model)
            # 2) AAE context
            metrics_aae = calculate_log_metrics(row, context_variant="AAE", model=model)

            # extract delta sae and delta aae
            delta_sae = metrics_sae['log_difference_SAE']
            delta_aae = metrics_aae['log_difference_AAE']

            combined_metrics = {
                'index': idx,
                **metrics_sae,
                **metrics_aae,
                'delta_SAE_minus_AAE': delta_sae - delta_aae
            }
            all_metrics.append(combined_metrics)

    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(all_metrics).set_index('index')

    # Merge metrics back into original DataFrame
    df_result = df_indexed.join(df_metrics, how='left')
    # Save results to output CSV
    df_result.to_csv(args.output, index=False)
    logger.info(f"Saved evaluation results to {args.output}")

if __name__ == "__main__":
    main()
        
