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
    elif model_name == 'llama31_8b':
        from models import Llama318BInterface
        return Llama318BInterface()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
def main():
    parser = argparse.ArgumentParser(description="Calculate dialect vector")
    parser.add_argument("--model", "-m", required=True,
                        choices=['gpt4o_mini', 'gpt41_batch', 'claude_haiku', 'phi3_medium', 'phi4_vllm', 'llama31_8b'],
                        help="Model to use for evaluation")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--sae_column', type=str, default='sae_text', help="Column name for SAE context")
    parser.add_argument('--aae_column', type=str, default='aae_text', help="Column name for AAE context")
    parser.add_argument('--api_key', type=str, default=None, help="API key for the model (if required)")
    parser.add_argument('--samples', type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument('--output', '-o', type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.input)

    # Validate columns
    for col in [args.sae_column, args.aae_column]:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in dataset")
            sys.exit(1)

    # Get model interface
    try:
        model = get_model_interface(args.model, api_key=args.api_key)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # ---------------------------
    # DEBUG: confirm imports/paths
    # ---------------------------
    import inspect
    import models as models_module
    logger.info(f"[DEBUG] models module file = {os.path.abspath(models_module.__file__)}")
    logger.info(f"[DEBUG] model class = {model.__class__}")
    logger.info(f"[DEBUG] model class file = {os.path.abspath(inspect.getfile(model.__class__))}")
    logger.info(f"[DEBUG] sweep method file = {os.path.abspath(inspect.getfile(model.layer_sweep_dialect_vectors_full.__func__))}")

    pairs = []
    for _, row in df.iterrows():
        sae_text = row[args.sae_column]
        aae_text = row[args.aae_column]
        pairs.append((sae_text, aae_text))

    if args.samples:
        pairs = pairs[:args.samples]

    logger.info(f"Processing {len(pairs)} text pairs for dialect vector calculation")
    logger.info(f"Running dialect vector calculation using model: {args.model}")

    results, best_layer = model.layer_sweep_dialect_vectors_full(
        pairs=pairs,
        max_pairs=args.samples,   # None => use all
    )

    # ---------------------------
    # DEBUG: confirm full sweep
    # ---------------------------
    layer_keys = sorted(list(results.keys()))
    logger.info(f"[DEBUG] results layers count = {len(layer_keys)}")
    logger.info(f"[DEBUG] results first keys = {layer_keys[:10]}")
    logger.info(f"[DEBUG] results last keys  = {layer_keys[-10:]}")
    if len(layer_keys) < 10:
        logger.warning("[DEBUG] Too few layers saved in results. This is NOT a full sweep output.")

    logger.info(f"Best layer identified: {best_layer}")

    # ---------------------------
    # Save outputs safely (avoid overwriting / GPU tensors)
    # ---------------------------
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # absolute output paths (avoid reading wrong file later)
    pt_path = os.path.abspath(f"{args.model}_dialect_vectors_full_{stamp}.pt")
    csv_path = os.path.abspath(args.output)

    # Convert to CPU + detach, and standardize key name to v_aae_unit
    safe_results = {}
    for lid, data in results.items():
        v = data.get("v_aae_unit", None)
        if v is None:
            v = data.get("v_aae", None)   # your interface uses "v_aae" but it's unit already
        if v is None:
            raise KeyError(f"Layer {lid} missing v_aae/v_aae_unit in results.")

        if torch.is_tensor(v):
            v = v.detach().float().cpu()

        safe_results[int(lid)] = {
            "v_aae_unit": v,
            "stats": data["stats"]
        }

    payload = {
        "model_name": args.model,
        "best_layer": int(best_layer),
        "results": safe_results
    }

    torch.save(payload, pt_path)
    logger.info(f"Saved dialect vector results (PT) to {pt_path}")
    logger.info(f"[DEBUG] PT saved layers={len(safe_results)}; keys head={sorted(safe_results)[:5]} tail={sorted(safe_results)[-5:]}")

    # Save CSV (from safe_results stats)
    rows = []
    for layer_id, data in safe_results.items():
        stats = data["stats"]
        rows.append({
            "layer_id": layer_id,
            "proj_mean_SAE": stats["proj_mean_SAE"],
            "proj_std_SAE": stats["proj_std_SAE"],
            "proj_mean_AAE": stats["proj_mean_AAE"],
            "proj_std_AAE": stats["proj_std_AAE"],
            "mean_gap_AAE_minus_SAE": stats["mean_gap_AAE_minus_SAE"],
            "is_best_layer": int(layer_id == int(best_layer)),
        })

    stats_df = pd.DataFrame(rows).sort_values(by="layer_id")
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"Saved dialect vector calculation results (CSV) to {csv_path}")


if __name__ == "__main__":
    main()