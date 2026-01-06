"""
Script to obtain sentiment classifications for AAE tweets using GPT-4.1 Batch API.
"""
import multiprocessing as mp
from ast import arg
from math import log
import os
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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aae_sentiment_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def _is_error_value(v: Any) -> bool:
    """Retrun True if a sentiment cell should be treated as ERROR/needs rerun."""
    if v is None:
        return True
    try:
        s = str(v).strip().lower()
    except Exception:
        return True
    return (s == '') or (s == 'nan') or (s == 'error') or (s == 'failed') or (s == 'err')

def load_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the AAE dataset from a CSV file.
    
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
    elif model_name == 'claude_batch':
        from models import ClaudeBatchInterface
        return ClaudeBatchInterface(api_key=api_key)
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
    elif model_name == 'phi4_hf':
        from models import Phi4HFModelInterface
        return Phi4HFModelInterface()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def analyze_sentiment_batch(
    texts: List[str],
    model_interface,
    beta: float = 0.0,
    layer_id: Optional[int] = None,
    v_aae_unit: Optional[torch.Tensor] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of texts.
    If (beta!=0 and v/layer provided) and model supports steering, use steered sentiment.
    """
    # --- steered path (Phi3) ---
    if beta != 0.0 and layer_id is not None and v_aae_unit is not None:
        # Prefer a batch steered API if you add it
        if hasattr(model_interface, "batch_get_sentiment_steered"):
            sentiments = model_interface.batch_get_sentiment_steered(
                texts=texts,
                layer_id=layer_id,
                v_aae_unit=v_aae_unit,
                beta=beta,
                batch_size=batch_size,
            )
        else:
            # Fallback: loop over single-call steered sentiment (slower but works)
            sentiments = [
                model_interface.get_sentiment_steered(
                    text=t,
                    layer_id=layer_id,
                    v_aae_unit=v_aae_unit,
                    beta=beta,
                )
                for t in texts
            ]
    else:
        # --- baseline path ---
        sentiments = model_interface.batch_get_sentiment(texts)

    for i, sentiment in enumerate(sentiments):
        sentiment["text"] = texts[i]
    return sentiments

def load_vector_from_pt(vector_pt: str, layer_id: int, key_prefer: str = "v_aae_unit") -> torch.Tensor:
    payload = torch.load(vector_pt, map_location="cpu")
    d = payload["results"][int(layer_id)]
    v = d.get(key_prefer, None)
    if v is None:
        v = d.get("v_aae", None)
    if v is None:
        raise KeyError(f"Vector not found in pt for layer={layer_id}. Keys={list(d.keys())}")
    if torch.is_tensor(v):
        v = v.detach().float().cpu()
    return v
def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment for AAE tweets using GPT-4.1 Batch API")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with AAE tweets")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file for sentiment results")
    parser.add_argument("--model", "-m", required=True, 
                        choices=['gpt4o_mini', 'gpt41_batch', 'claude_haiku', 'claude_batch', 'phi3_medium', 'phi4_vllm', "mistral31_vllm", "gemma3_vllm", "deepseekR1_vllm", "phi4_hf"], 
                        help="Model to use for sentiment analysis")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--api-key", "-k", help="API key for the selected model (if applicable)")
    parser.add_argument("--text-column", "-t", default="aae_text", help="Column name containing the text to analyze")
    parser.add_argument("--beta", type=float, default=0.0, help="Steering strength beta (phi3 only). 0 = baseline.")
    parser.add_argument("--layer", type=int, default=38, help="Steering layer id (phi3 only).")
    parser.add_argument("--vector-pt", type=str, default=None, help="Path to dialect vector pt file (phi3 only).")
    parser.add_argument("--vector-key", type=str, default="v_aae_unit", help="Vector key to load from pt (v_aae_unit or v_aae).")
    parser.add_argument("--vector-scale", type=float, default=1000, help="Optional scaling applied to loaded vector.")

    args = parser.parse_args()
    
    if args.beta != 0.0:
        root, ext = os.path.splitext(args.output)
        args.output = f"{root}_beta{args.beta}_layer{args.layer}{ext}"
    # Load the dataset
    df = load_dataset(args.input, args.samples)
    
    # Get the model interface
    try:
        model_interface = get_model_interface(args.model, args.api_key)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    v_aae_unit = None
    layer_id = None

    if args.model == "phi3_medium" and args.beta != 0.0:
        if not args.vector_pt:
            raise ValueError("--vector-pt is required when --beta != 0 for phi3_medium")
        layer_id = int(args.layer)
        v_aae_unit = load_vector_from_pt(args.vector_pt, layer_id, key_prefer=args.vector_key)
        v_aae_unit = v_aae_unit * float(args.vector_scale)
        logger.info(f"[STEER] phi3_medium beta={args.beta}, layer={layer_id}, vector_scale={args.vector_scale}, norm={v_aae_unit.norm().item():.6f}")
    elif args.model == "phi4_hf" and args.beta != 0.0:
        if not args.vector_pt:
            raise ValueError("--vector-pt is required when --beta != 0 for phi4_hf")
        layer_id = int(args.layer)
        v_aae_unit = load_vector_from_pt(args.vector_pt, layer_id, key_prefer=args.vector_key)
        v_aae_unit = v_aae_unit * float(args.vector_scale)
        logger.info(f"[STEER] phi4_hf beta={args.beta}, layer={layer_id}, vector_scale={args.vector_scale}, norm={v_aae_unit.norm().item():.6f}")
    else:
        logger.info(f"[BASELINE] model={args.model}, beta={args.beta} (steering disabled)")

    # Extract texts to analyze
    if args.text_column in df.columns:
        texts = df[args.text_column].tolist()
    else:
        logger.error(f"Input CSV must have '{args.text_column}' column with text to analyze")
        sys.exit(1)

    if 'sentiment' in df.columns:
        # Filter to only rows needing sentiment analysis
        mask_to_process = df['sentiment'].apply(_is_error_value)
        num_already_ok = (~mask_to_process).sum()
        num_to_process = mask_to_process.sum()
        logger.info(f"{num_already_ok} records already have sentiment, processing {num_to_process} records")
        
        indices_to_process = df.index[mask_to_process].tolist()
        texts_to_process = df.loc[indices_to_process, args.text_column].tolist()

        processed_sentiments: List[Dict[str, Any]] = []
        for start in tqdm(range(0, len(texts_to_process), args.batch_size), desc="Analyzing sentiment"):
            batch_texts = texts_to_process[start:start+args.batch_size]
            try:
                sentiments = analyze_sentiment_batch(
                    batch_texts,
                    model_interface,
                    beta=args.beta,
                    layer_id=layer_id,
                    v_aae_unit=v_aae_unit,
                    batch_size=args.batch_size if args.batch_size <= 64 else 64,  # optional safety
                )
                processed_sentiments.extend(sentiments)
            except Exception as e:
                logger.error(f"Error processing batch starting at index {start}: {str(e)}")
                # Add placeholder for failed analyses
                failed_sentiments = [
                    {
                        'text': text,
                        'sentiment': 'ERROR'
                    } for text in batch_texts
                ]
                processed_sentiments.extend(failed_sentiments)
            
            # Clean up GPU memory if using Phi models
            if (args.model == 'phi3_medium' or args.model == 'phi4_vllm') and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(processed_sentiments) != num_to_process:
            logger.error(f"Expected {num_to_process} processed sentiments, got {len(processed_sentiments)}")
            
        for idx, result in zip(indices_to_process, processed_sentiments):
            df.at[idx, 'sentiment'] = result.get('sentiment', 'ERROR')
        
        df.to_csv(args.output, index=False)
        logger.info(f"Saved updated dataset with sentiment to {args.output}")

        sentiment_counts = df['sentiment'].value_counts(dropna=False)
        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")

        original_df = pd.read_csv(args.input)
        if 'sentiment' in original_df.columns:
            old_error_count = original_df['sentiment'].apply(_is_error_value).sum()
        else:
            old_error_count = 0

        new_error_count = int(df['sentiment'].astype(str).str.strip().str.lower().eq('error').sum())
        fixed_count = old_error_count - new_error_count if old_error_count >= new_error_count else 0

        logger.info(f"Errors before re-run: {old_error_count}, after re-run: {new_error_count}, fixed: {fixed_count}")
        
    else:
        # Analyze sentiment in batches
        all_sentiments = []
        # Process batches
        for i in tqdm(range(0, len(df), args.batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i+args.batch_size]
            
            try:
                sentiments = analyze_sentiment_batch(
                    batch_texts,
                    model_interface,
                    beta=args.beta,
                    layer_id=layer_id,
                    v_aae_unit=v_aae_unit,
                    batch_size=args.batch_size if args.batch_size <= 64 else 64,  # optional safety
                )
                all_sentiments.extend(sentiments)
                
                # Log progress
                if (i // args.batch_size) % 10 == 0:
                    logger.info(f"Processed {i+len(sentiments)}/{len(df)} records")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//args.batch_size}: {str(e)}")
                # Add placeholder for failed analyses
                failed_sentiments = [
                    {
                        'text': text,
                        'sentiment': 'ERROR',
                        'score': 0,
                        'raw_response': f"Batch processing error: {str(e)}"
                    } for text in batch_texts
                ]
                all_sentiments.extend(failed_sentiments)
            
            # Clean up GPU memory if using Phi models
            if (args.model == 'phi3_medium' or args.model == 'phi4_vllm') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Create the output DataFrame
        output_df = pd.DataFrame({
            'aae_text': [s['text'] for s in all_sentiments],
            'sentiment': [s['sentiment'] for s in all_sentiments]
        })
        
        # Save the output
        output_df.to_csv(args.output, index=False)
        logger.info(f"Saved {len(output_df)} sentiment analyses to {args.output}")
        
        # Print some statistics
        sentiment_counts = output_df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
        
        error_count = sum(1 for s in output_df['sentiment'] if s == 'ERROR')
        if error_count > 0:
            logger.warning(f"{error_count} records had errors during processing")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()