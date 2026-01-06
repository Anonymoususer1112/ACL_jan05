import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
from ast import arg, parse
import os

from pyexpat import model
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
import numpy as np
from zmq import has
from vllm import LLM

# Change relative import to absolute import
try:
    from .model_interface import  ModelInterface
except ImportError:
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from models.model_interface import ModelInterface
    from models import Phi4VllmInterface, Mistral31VllmInterface, Gemma3VllmInterface, DeepseekR1VllmInterface

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("preference_multiple_choice.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

class DialectEvaluator(ModelInterface):
    def __init__(self, model_id: str, api_key: Optional[str] = None, dtype: str = "float16" ):
        super().__init__("dialect_evaluator")
        self.model_id = model_id
        self.api_key = api_key
        self.dtype = dtype

        if model_id == "phi4_vllm":
            self.model = Phi4VllmInterface(model_id="microsoft/phi-4", dtype=dtype)
        elif model_id == "mistral31_vllm":
            self.model = Mistral31VllmInterface(model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503", dtype=dtype)
        elif model_id == "gemma3_vllm":
            self.model = Gemma3VllmInterface(model_id="google/gemma-3-27b-it", dtype=dtype)
        elif model_id == "deepseekR1_vllm":
            self.model = DeepseekR1VllmInterface(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", dtype=dtype)
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")
        
        self.sampling_params = self.model.sampling_params
        self.embed_model = None

        logger.info(f"Initialized DialectEvaluator with model {self.model_id}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for input texts using Mistral-3.1 with vLLM.
        
        This method uses vLLM's embedding API to get text embeddings.
         str) ->
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        
        # Create a new LLM instance specifically for embeddings
        # We need to reinitialize with task="embed" to use embedding functionality
        logger.info("Initializing embedding model...")
        
        # Use the model_id and dtype from our class
        model_id = self.model_id
        dtype = self.dtype
        
        try:
            # Initialize embedding model with vLLM's embedding task
            if self.embed_model is None:
                self.embed_model = LLM(
                    model=model_id,
                    dtype=dtype,
                    task="embed",
                    enforce_eager=True,
                    gpu_memory_utilization=0.5
                )
            embed_model = self.embed_model
            
            embeddings_list = []
            batch_size = 16  # Process in batches to avoid OOM issues
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
                batch_texts = texts[i:i + batch_size]

                # Generate embeddings for the batch
                outputs = embed_model.embed(batch_texts)
                
                # Extract embeddings from outputs
                for output in outputs:
                    embedding = np.array(output.outputs.embedding)
                    embeddings_list.append(embedding)
                
            # Stack all embeddings
            return np.vstack(embeddings_list)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def get_log_probs(
        self,
        contexts: List[str],
        continuations: List[str],
        batch_size: int = 16,
        separator: str = "\n",
    ) -> List[Dict[str, float]]:
        import math
        from vllm import SamplingParams

        llm = self.model.llm
        tok = llm.get_tokenizer()

        # IMPORTANT: request PROMPT logprobs (not generation logprobs)
        score_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,        
            prompt_logprobs=1,   
            logprobs=0           # optional: don't care about generated token logprobs
        )

        full_texts, starts, ends, n_ctx_tokens_list = [], [], [], []

        for ctx_raw, cont_raw in zip(contexts, continuations):
            ctx = str(ctx_raw).rstrip()
            cont = str(cont_raw).lstrip()

            ctx_for_boundary = ctx + separator
            full_text = ctx_for_boundary + cont

            ctx_ids = tok.encode(ctx_for_boundary, add_special_tokens=False)
            full_ids = tok.encode(full_text, add_special_tokens=False)

            full_texts.append(full_text)
            n_ctx_tokens_list.append(len(ctx_ids))
            starts.append(len(ctx_ids))
            ends.append(len(full_ids))

        results: List[Dict[str, float]] = []

        for i in tqdm(
            range(0, len(full_texts), batch_size),
            desc="Calculating PPL(cont|ctx)"
        ):
            batch_full = full_texts[i:i + batch_size]
            batch_outs = llm.generate(batch_full, score_params)

            for j, out in enumerate(batch_outs):
                idx = i + j

                # vLLM returns prompt_logprobs on the RequestOutput (NOT on out.outputs[0])
                prompt_lp = getattr(out, "prompt_logprobs", None)
                if prompt_lp is None:
                    raise RuntimeError(
                        "vLLM did not return prompt_logprobs. "
                        "Your vLLM version may not support prompt_logprobs."
                    )

                # prompt_lp is a list aligned with prompt tokens; elements can be None or dict(token_id -> Logprob)
                # We need logprob of the actually observed token at each position.
                token_logprobs = []
                for item in prompt_lp:
                    if item is None:
                        token_logprobs.append(None)
                        continue
                    if isinstance(item, dict) and len(item) > 0:
                        token_logprobs.append(next(iter(item.values())).logprob)
                    else:
                        # fallback if item is already a Logprob-like object
                        token_logprobs.append(getattr(item, "logprob", None))

                s, e = starts[idx], ends[idx]
                cont_slice = token_logprobs[s:e]

                # drop None (some implementations put None for first token)
                cont_logprobs = [x for x in cont_slice if x is not None]

                n_ctx_tokens = n_ctx_tokens_list[idx]
                n_cont_tokens = len(cont_logprobs)

                if n_cont_tokens == 0:
                    results.append({
                        "n_ctx_tokens": float(n_ctx_tokens),
                        "n_cont_tokens": 0.0,
                        "avg_nll_cont": float("inf"),
                        "log_ppl_cont": float("inf"),
                        "ppl_cont": float("inf"),
                    })
                    continue

                avg_nll = -sum(cont_logprobs) / n_cont_tokens
                ppl = math.exp(avg_nll)

                results.append({
                    "n_ctx_tokens": float(n_ctx_tokens),
                    "n_cont_tokens": float(n_cont_tokens),
                    "avg_nll_cont": float(avg_nll),
                    "log_ppl_cont": float(avg_nll),
                    "ppl_cont": float(ppl),
                })

        return results

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

    
def main():
    parser = argparse.ArgumentParser(description="Preference Multiple Choice Task Evaluation")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model", "-m", required=True, 
                        choices=['gpt4o_mini', 'gpt41_batch', 'claude_haiku', 'phi3_medium', 'phi4_vllm', "mistral31_vllm", "gemma3_vllm", "deepseekR1_vllm"], 
                        help="Model to use for evaluation")
    parser.add_argument("--dtype", type=str, default="float16", help="Model dtype (default: float16)")

    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--api-key", "-k", help="API key for the selected model (if applicable)")
    parser.add_argument("--sae-context-column", default="sae_context", help="Column name containing SAE context")
    parser.add_argument("--aae-context-column", default="aae_context", help="Column name containing AAE context")
    parser.add_argument("--sae-cont-column", default="sae_continuation", help="Column name containing SAE continuation")
    parser.add_argument("--aae-cont-column", default="aae_continuation", help="Column name containing AAE continuation")
    parser.add_argument("--separator", type=str, default="\n", help="Separator between context and continuation")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.input, num_samples=args.samples)

    # Validate columns 
    for col in [args.sae_context_column, args.aae_context_column,
                args.sae_cont_column, args.aae_cont_column]:
        if col not in df.columns:
            logger.error(f"Specified column '{col}' not found in dataset")
            sys.exit(1)

    # Build evaluator
    evaluator = DialectEvaluator(model_id=args.model, dtype=args.dtype)

    # Prepare canonical columns we will output
    df = df.reset_index(drop=True).copy()

    df["aae_context"] = df[args.aae_context_column].astype(str)
    df["sae_context"] = df[args.sae_context_column].astype(str)
    df["aae_continuation"] = df[args.aae_cont_column].astype(str)
    df["sae_continuation"] = df[args.sae_cont_column].astype(str)

    # Storage for metrics
    log_prob_aae_SAE = [None] * len(df)
    log_prob_sae_SAE = [None] * len(df)
    log_diff_SAE     = [None] * len(df)

    log_prob_aae_AAE = [None] * len(df)
    log_prob_sae_AAE = [None] * len(df)
    log_diff_AAE     = [None] * len(df)

    def to_logp(stats: Dict[str, float]) -> float:
        """
        Convert get_log_probs() output to total logP(continuation | context).

        stats contains:
          - avg_nll_cont = - mean(logprob over continuation tokens)
          - n_cont_tokens
        => sum logprob = -avg_nll_cont * n_cont_tokens
        """
        n = float(stats.get("n_cont_tokens", 0.0))
        avg_nll = float(stats.get("avg_nll_cont", float("inf")))
        if n <= 0 or not (avg_nll < float("inf")):
            return float("-inf")
        return -avg_nll * n

    total = len(df)
    logger.info(f"Scoring {total} rows with model={args.model}, batch_size={args.batch_size}")

    # Batch loop
    indices = list(range(total))
    for batch_ids in tqdm(list(chunked_iter(indices, args.batch_size)), desc="Batches"):
        # gather texts for this batch
        sae_ctx  = df.loc[batch_ids, "sae_context"].tolist()
        aae_ctx  = df.loc[batch_ids, "aae_context"].tolist()
        sae_cont = df.loc[batch_ids, "sae_continuation"].tolist()
        aae_cont = df.loc[batch_ids, "aae_continuation"].tolist()

        # 4 combinations (vectorized per batch):
        # 1) SAE ctx + AAE cont
        stats_aae_SAE = evaluator.get_log_probs(
            contexts=sae_ctx,
            continuations=aae_cont,
            batch_size=min(args.batch_size, 64),
            separator=args.separator,
        )
        # 2) SAE ctx + SAE cont
        stats_sae_SAE = evaluator.get_log_probs(
            contexts=sae_ctx,
            continuations=sae_cont,
            batch_size=min(args.batch_size, 64),
            separator=args.separator,
        )
        # 3) AAE ctx + AAE cont
        stats_aae_AAE = evaluator.get_log_probs(
            contexts=aae_ctx,
            continuations=aae_cont,
            batch_size=min(args.batch_size, 64),
            separator=args.separator,
        )
        # 4) AAE ctx + SAE cont
        stats_sae_AAE = evaluator.get_log_probs(
            contexts=aae_ctx,
            continuations=sae_cont,
            batch_size=min(args.batch_size, 64),
            separator=args.separator,
        )

        # write back row-by-row
        for k, ridx in enumerate(batch_ids):
            lp_aae_SAE = to_logp(stats_aae_SAE[k])
            lp_sae_SAE = to_logp(stats_sae_SAE[k])
            lp_aae_AAE = to_logp(stats_aae_AAE[k])
            lp_sae_AAE = to_logp(stats_sae_AAE[k])

            log_prob_aae_SAE[ridx] = lp_aae_SAE
            log_prob_sae_SAE[ridx] = lp_sae_SAE
            log_diff_SAE[ridx]     = lp_sae_SAE - lp_aae_SAE

            log_prob_aae_AAE[ridx] = lp_aae_AAE
            log_prob_sae_AAE[ridx] = lp_sae_AAE
            log_diff_AAE[ridx]     = lp_sae_AAE - lp_aae_AAE

    # Attach metrics
    df["log_prob_aae_SAE"] = log_prob_aae_SAE
    df["log_prob_sae_SAE"] = log_prob_sae_SAE
    df["log_difference_SAE"] = log_diff_SAE

    df["log_prob_aae_AAE"] = log_prob_aae_AAE
    df["log_prob_sae_AAE"] = log_prob_sae_AAE
    df["log_difference_AAE"] = log_diff_AAE

    # Output exactly the required columns (in order)
    out_cols = [
        "aae_text",
        "sae_text",
        "aae_context",
        "aae_continuation",
        "sae_context",
        "sae_continuation",
        "log_prob_aae_SAE",
        "log_prob_sae_SAE",
        "log_difference_SAE",
        "log_prob_aae_AAE",
        "log_prob_sae_AAE",
        "log_difference_AAE",
    ]
    df[out_cols].to_csv(args.output, index=False)
    logger.info(f"Saved results to {args.output} with columns={out_cols}")
if __name__ == "__main__":
    main()
        
