"""
Script for preference multiple choice tasks.
"""

from ast import arg, parse
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import random
import sys
import csv
from banal import chunked_iter
import pandas as pd
import logging
import argparse
import asyncio
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

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
    elif model_name == 'mistral31_vllm':
        from models import Mistral31VllmInterface
        return Mistral31VllmInterface()
    elif model_name == 'gemma3_vllm':
        from models import Gemma3VllmInterface
        return Gemma3VllmInterface()
    elif model_name == 'deepseekR1_vllm':
        from models import DeepseekR1VllmInterface
        return DeepseekR1VllmInterface()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

    
def main():
    parser = argparse.ArgumentParser(description="Preference Multiple Choice Task Evaluation")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model", "-m", required=True, 
                        choices=['gpt4o_mini', 'gpt41_batch', 'claude_haiku', 'phi3_medium', 'phi4_vllm', "mistral31_vllm", "gemma3_vllm", "deepseekR1_vllm"], 
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

    rows = df[[args.sae_context_column,
               args.aae_context_column,
               args.sae_cont_column,
               args.aae_cont_column]].to_dict(orient='records')

    total = len(rows)
    logger.info(f"Loaded {total} records for evaluation")

    outputs_sae: List[Dict[str, Any]] = []
    outputs_aae: List[Dict[str, Any]] = []

    for batch in tqdm(chunked_iter(list(enumerate(rows)), args.batch_size),
                    total=(total + args.batch_size - 1) // args.batch_size):

        for idx, row in batch:
            context_sae = str(row[args.sae_context_column]).strip()
            context_aae = str(row[args.aae_context_column]).strip()
            cont_sae = str(row[args.sae_cont_column]).strip()
            cont_aae = str(row[args.aae_cont_column]).strip()

            rng = random.Random(idx)  # Seed for reproducibility

            # -------- 1) SAE context --------
            try:
                pref_sae = model.get_continuation_preference(
                    context_sae=context_sae,
                    continuation_sae=cont_sae,
                    context_aae=context_aae,
                    continuation_aae=cont_aae,
                    context_variant="SAE",
                    rng=rng,
                )

                outputs_sae.append({
                    'index': idx,
                    'context_variant': "SAE",
                    'sae_context': context_sae,
                    'aae_context': context_aae,
                    'sae_continuation': cont_sae,
                    'aae_continuation': cont_aae,
                    'preferred': pref_sae['preferred'],
                    'A_is': pref_sae['A_is'],
                    'B_is': pref_sae['B_is'],
                    'raw_response': pref_sae.get('raw_response', '')
                })
            except Exception as e:
                logger.error(f"[SAE context] Error at index {idx}: {e}")
                outputs_sae.append({
                    'index': idx,
                    'context_variant': "SAE",
                    'sae_context': context_sae,
                    'aae_context': context_aae,
                    'sae_continuation': cont_sae,
                    'aae_continuation': cont_aae,
                    'preferred': 'ERROR',
                    'A_is': 'ERROR',
                    'B_is': 'ERROR',
                    'raw_response': '',
                })

            # -------- 2) AAE context --------
            try:
                pref_aae = model.get_continuation_preference(
                    context_sae=context_sae,
                    continuation_sae=cont_sae,
                    context_aae=context_aae,
                    continuation_aae=cont_aae,
                    context_variant="AAE",
                    rng=rng,
                )

                outputs_aae.append({
                    'index': idx,
                    'context_variant': "AAE",
                    'sae_context': context_sae,
                    'aae_context': context_aae,
                    'sae_continuation': cont_sae,
                    'aae_continuation': cont_aae,
                    'preferred': pref_aae['preferred'],
                    'A_is': pref_aae['A_is'],
                    'B_is': pref_aae['B_is'],
                    'raw_response': pref_aae.get('raw_response', '')
                })
            except Exception as e:
                logger.error(f"[AAE context] Error at index {idx}: {e}")
                outputs_aae.append({
                    'index': idx,
                    'context_variant': "AAE",
                    'sae_context': context_sae,
                    'aae_context': context_aae,
                    'sae_continuation': cont_sae,
                    'aae_continuation': cont_aae,
                    'preferred': 'ERROR',
                    'A_is': 'ERROR',
                    'B_is': 'ERROR',
                    'raw_response': '',
                })


    # Save results to two output CSVs
    base, ext = os.path.splitext(args.output)
    if ext == "":
        ext = ".csv"

    sae_path = base + "_sae" + ext
    aae_path = base + "_aae" + ext

    df_sae = pd.DataFrame(outputs_sae)
    df_aae = pd.DataFrame(outputs_aae)

    df_sae.to_csv(sae_path, index=False)
    df_aae.to_csv(aae_path, index=False)

    logger.info(f"Saved SAE-context results to {sae_path}")
    logger.info(f"Saved AAE-context results to {aae_path}")

    for label, df_out in [("SAE", df_sae), ("AAE", df_aae)]:
        total = len(df_out)
        error_count = (df_out['preferred'] == 'ERROR').sum()
        sae_pref = (df_out['preferred'] == 'SAE').sum()
        aae_pref = (df_out['preferred'] == 'AAE').sum()
        logger.info(f"[{label} context] Total: {total}, "
                    f"Errors: {error_count} ({(error_count/total)*100:.2f}%), "
                    f"SAE preferred: {sae_pref} ({(sae_pref/total)*100:.2f}%), "
                    f"AAE preferred: {aae_pref} ({(aae_pref/total)*100:.2f}%)")


if __name__ == "__main__":
    main()
                


    
