from ast import arg, parse
import os

import random
import sys
import csv
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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

def parse_betas(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser("Phi-3-medium SAE corpus PPL sweep with activation steering")
    p.add_argument("--input_csv", required=True, help="CSV that contains SAE texts")
    p.add_argument("--sae_col", default="sae_text", help="Column name for SAE text")
    p.add_argument("--vector_pt", required=True, help="Path to saved dialect vectors .pt")
    p.add_argument("--layer", type=int, default=38, help="Layer id to steer (default: 38)")
    p.add_argument("--betas", default="0, 0.2, 0.4, 0.6, 0.8, 1.0",
                   help="Comma-separated betas")
    p.add_argument("--max_rows", type=int, default=None, help="Use only first N rows (debug)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for PPL")
    p.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length")
    p.add_argument("--out_csv", required=True, help="Output CSV for sweep results")
    p.add_argument("--scale", type=float, default=1000.0,
               help="Scale factor for dialect vector (v <- scale * v)")
    args = p.parse_args()

    # ---- load SAE corpus ----
    df = pd.read_csv(args.input_csv)
    if args.sae_col not in df.columns:
        raise ValueError(f"Column '{args.sae_col}' not found. Columns={list(df.columns)}")

    texts = df[args.sae_col].astype(str).tolist()
    if args.max_rows is not None:
        texts = texts[:args.max_rows]

    logger.info(f"Loaded SAE texts: {len(texts)} rows from {args.input_csv}")

    # ---- load dialect vector ----
    payload = torch.load(args.vector_pt, map_location="cpu")
    layer_id = int(args.layer)
    if "results" not in payload or layer_id not in payload["results"]:
        raise KeyError(f"Layer {layer_id} not found in payload['results']. Available layers: {sorted(payload.get('results', {}).keys())[:10]}...")

    v = payload["results"][layer_id]["v_aae"]

    v = v * float(args.scale)

    logger.info(
        f"Loaded dialect vector: layer={args.layer}, "
        f"norm(unit)~1.0, scale={args.scale}, "
        f"effective_norm={v.norm().item():.6f}"
    )
    

    logger.info(f"Loaded v_aae_unit from {args.vector_pt} at layer={layer_id}, dim={tuple(v.shape)}")

    model = get_model_interface("phi3_medium")

    betas = parse_betas(args.betas)
    rows = []
    for beta in betas:
        logger.info(f"[SWEEP] beta={beta} layer={layer_id}")
        if beta == 0.0:
            metrics = model.corpus_ppl(
                texts=texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                beta=0.0
            )
        else:
            metrics = model.corpus_ppl(
                texts=texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                layer_id=layer_id,
                v_aae_unit=v,
                beta=beta
            )

        rows.append({
            "model": "phi3_medium",
            "layer": layer_id,
            **metrics
        })

        # free VRAM between runs
        torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows).sort_values(["layer", "beta"])
    out_df.to_csv(args.out_csv, index=False)
    logger.info(f"Saved sweep results -> {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()