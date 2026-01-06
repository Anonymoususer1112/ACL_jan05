import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from vllm import LLM
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dialect_evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

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

class Mistral31DialectEvaluator(ModelInterface):
    def __init__(self, model_id: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503", dtype: str = "float16", device: str = "cuda"):
        """
        Initialize the Mistral-3.1 evaluator.
        
        Args:
            model_id: HuggingFace model ID for Mistral-3.1
            dtype: Data type for model weights (bfloat16, float16, etc.)
            device: Device to run model on (cuda or cpu)
        """
        super().__init__("mistral31_dialect_eval")
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        
        # Initialize the Mistral31VllmInterface
        self.model = Mistral31VllmInterface(model_id=model_id, dtype=dtype)
        
        # Store the sampling params for get_log_probs
        self.sampling_params = self.model.sampling_params
        self.embed_model = None
        
        logger.info(f"Initialized Mistral31DialectEvaluator with model {model_id}")

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

class Gemma3DialectEvaluator(ModelInterface):
    def __init__(self, model_id: str = "google/gemma-3-27b-it", dtype: str = "float16", device: str = "cuda"):
        """
        Initialize the Gemma-3 evaluator.
        
        Args:
            model_id: HuggingFace model ID for Gemma-3
            dtype: Data type for model weights (bfloat16, float16, etc.)
            device: Device to run model on (cuda or cpu)
        """
        super().__init__("gemma3_dialect_eval")
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        
        # Initialize the Gemma3VllmInterface
        self.model = Gemma3VllmInterface(model_id=model_id, dtype=dtype)
        
        # Store the sampling params for get_log_probs
        self.sampling_params = self.model.sampling_params
        self.embed_model = None
        
        logger.info(f"Initialized Gemma3DialectEvaluator with model {model_id}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
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

class Phi4DialectEvaluator(ModelInterface):
    def __init__(self, model_id: str = "microsoft/phi-4", dtype: str = "bfloat16", device: str = "cuda"):
        """
        Initialize the Phi4 evaluator.
        
        Args:
            model_id: HuggingFace model ID for Phi-4
            dtype: Data type for model weights (bfloat16, float16, etc.)
            device: Device to run model on (cuda or cpu)
        """
        super().__init__("phi4_dialect_eval")
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        
        # Initialize the Phi4VllmInterface
        self.model = Phi4VllmInterface(model_id=model_id, dtype=dtype)
        
        # Store the sampling params for get_log_probs
        self.sampling_params = self.model.sampling_params
        self.embed_model = None
        
        logger.info(f"Initialized Phi4DialectEvaluator with model {model_id}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for input texts using Phi4 with vLLM.
        
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
    
class DeepseekR1DialectEvaluator(ModelInterface):
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", dtype: str = "float16", device: str = "cuda"):
        """
        Initialize the Deepseek-R1 evaluator.
        
        Args:
            model_id: HuggingFace model ID for Deepseek-R1
            dtype: Data type for model weights (bfloat16, float16, etc.)
            device: Device to run model on (cuda or cpu)
        """
        super().__init__("deepseekR1_dialect_eval")
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        
        # Initialize the DeepseekR1VllmInterface
        self.model = DeepseekR1VllmInterface(model_id=model_id, dtype=dtype)
        
        # Store the sampling params for get_log_probs
        self.sampling_params = self.model.sampling_params
        self.embed_model = None
        
        logger.info(f"Initialized DeepseekR1DialectEvaluator with model {model_id}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for input texts using DeepSeek R1 with vLLM.
        
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
        
class DialectEvaluator:
    """Evaluates dialect bias in language models by comparing SAE and AAE texts."""
    
    def __init__(
        self, 
        model: ModelInterface,
        output_dir: str = "output_evaluations/dialect_eval_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator with model and output settings.
        
        Args:
            model: Model implementing the ModelInterface
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Results will be saved to {self.output_dir}")
    
    def calculate_js_distance(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon distance between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            JS distance value
        """
        # Calculate mean embeddings
        mean_emb1 = np.mean(embeddings1, axis=0)
        mean_emb2 = np.mean(embeddings2, axis=0)
        
        # Normalize
        mean_emb1 = mean_emb1 / np.linalg.norm(mean_emb1)
        mean_emb2 = mean_emb2 / np.linalg.norm(mean_emb2)
        
        # Jensen-Shannon distance
        return jensenshannon(mean_emb1, mean_emb2)
    
    def visualize_embeddings(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray, 
        labels: Tuple[str, str] = ("Standard American English", "African American English"),
        n_samples: int = 500,
        filename: str = "dialect_embeddings_tsne.png"
    ) -> str:
        """
        Visualize embeddings using t-SNE and save the plot.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Labels for the two embedding sets
            n_samples: Max number of samples to visualize
            filename: Output filename for the plot
            
        Returns:
            Path to the saved visualization
        """
        # Sample if there are too many points
        if len(embeddings1) > n_samples:
            idx = np.random.choice(len(embeddings1), n_samples, replace=False)
            emb1_sample = embeddings1[idx]
            emb2_sample = embeddings2[idx]
        else:
            emb1_sample = embeddings1
            emb2_sample = embeddings2
        
        # Combine embeddings for t-SNE
        combined_embeddings = np.vstack([emb1_sample, emb2_sample])
        
        # Apply t-SNE
        logger.info("Applying t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
        transformed = tsne.fit_transform(combined_embeddings)
        
        # Split back into two groups
        transformed1 = transformed[:len(emb1_sample)]
        transformed2 = transformed[len(emb1_sample):]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(transformed1[:, 0], transformed1[:, 1], c='blue', label=labels[0], alpha=0.5)
        plt.scatter(transformed2[:, 0], transformed2[:, 1], c='red', label=labels[1], alpha=0.5)
        plt.legend()
        plt.title('t-SNE visualization of embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def evaluate(
            self,
            data_file: str,
            sae_context_col: str,
            aae_context_col: str,
            sae_cont_col: str,
            aae_cont_col: str,
            sample_size: Optional[int] = None,
            batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Evaluate dialect bias using the provided dataset.
        
        Args:
            data_file: Path to CSV file containing evaluation data
            sae_context_col: Column name for SAE context
            aae_context_col: Column name for AAE context
            sae_continuation_col: Column name for SAE continuation
            aae_continuation_col: Column name for AAE continuation
            sample_size: Number of samples to evaluate (None for all)
            batch_size: Batch size for model inference
            
        Returns:
            Dictionary of evaluation results
        """
        # Load dataset
        df = pd.read_csv(data_file)
        
        if sample_size is not None:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} records for evaluation.")
        
        sae_ctx = df[sae_context_col].astype(str).tolist()
        aae_ctx = df[aae_context_col].astype(str).tolist()
        sae_cont = df[sae_cont_col].astype(str).tolist()
        aae_cont = df[aae_cont_col].astype(str).tolist()

        logger.info(f"Analyzing {len(df)} examples -> 4 combos each...")
        
        # Long-format rows (4N)
        rows = []
        contexts_for_scoring = []
        conts_for_scoring = []

        for i in range(len(df)):
            combos = [
                ("SAE", "SAE", sae_ctx[i], sae_cont[i]),
                ("SAE", "AAE", sae_ctx[i], aae_cont[i]),
                ("AAE", "SAE", aae_ctx[i], sae_cont[i]),
                ("AAE", "AAE", aae_ctx[i], aae_cont[i]),
            ]
            for ctx_d, cont_d, ctx_text, cont_text in combos:
                rows.append({
                    "row_id": i,
                    "context_dialect": ctx_d,
                    "continuation_dialect": cont_d,
                    "combo": f"{ctx_d}_{cont_d}",
                    "context_text": ctx_text,
                    "continuation_text": cont_text,
                })
                contexts_for_scoring.append(ctx_text)
                conts_for_scoring.append(cont_text)

        long_df = pd.DataFrame(rows)

        # ---- Conditional Perplexity: PPL(continuation | context) ----
        logger.info("2. Calculating conditional perplexity PPL(cont|ctx) with vLLM logprobs...")
        score_dicts = self.model.get_log_probs(
            contexts_for_scoring,
            conts_for_scoring,
            batch_size=batch_size,
            separator="\n"   # keep consistent boundary
        )

        # Attach scores
        long_df["n_ctx_tokens"] = [d["n_ctx_tokens"] for d in score_dicts]
        long_df["n_cont_tokens"] = [d["n_cont_tokens"] for d in score_dicts]
        long_df["avg_nll_cont"] = [d["avg_nll_cont"] for d in score_dicts]
        long_df["log_ppl_cont"] = [d["log_ppl_cont"] for d in score_dicts]
        long_df["ppl_cont"] = [d["ppl_cont"] for d in score_dicts]

        # ---- Save one combined file (recommended) ----
        ppl_long_path = self.output_dir / "ppl_scores_long.csv"
        long_df.to_csv(ppl_long_path, index=False, encoding="utf-8")
        logger.info(f"Saved long-format PPL results to {ppl_long_path}")

        # ---- Summary by combo (optional but useful) ----
        summary = (long_df
                .groupby("combo")
                .agg(
                    n=("ppl_cont", "count"),
                    mean_log_ppl=("log_ppl_cont", "mean"),
                    std_log_ppl=("log_ppl_cont", "std"),
                    mean_ppl=("ppl_cont", "mean"),
                    std_ppl=("ppl_cont", "std"),
                    mean_n_ctx_tokens=("n_ctx_tokens", "mean"),
                    mean_n_cont_tokens=("n_cont_tokens", "mean"),
                )
                .reset_index()
                .sort_values("combo"))

        ppl_summary_path = self.output_dir / "ppl_summary_by_combo.csv"
        summary.to_csv(ppl_summary_path, index=False, encoding="utf-8")
        logger.info(f"Saved PPL summary to {ppl_summary_path}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate dialect bias using embedding space and perplexity analysis")
    
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to CSV file with paired dialect texts")
    parser.add_argument("--sae_context", type=str, default="sae_context",
                        help="Column name for Standard American English contexts")
    parser.add_argument("--aae_context", type=str, default="aae_context",
                        help="Column name for African American English contexts")
    parser.add_argument("--sae_continuation", type=str, default="sae_continuation",
                        help="Column name for Standard American English continuations")
    parser.add_argument("--aae_continuation", type=str, default="aae_continuation",
                        help="Column name for African American English continuations")
    parser.add_argument("--model", type=str, required=True, choices=["deepseekr1_vllm", "mistral31_vllm", "phi4_vllm", "gemma3_vllm"],
                        help="model name or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (bfloat16, float16)")
    parser.add_argument("--output_dir", type=str, default="output_evaluations/phi4_dialect_eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on ('cuda' or 'cpu')")
    
    args = parser.parse_args()

    if args.model == "mistral31_vllm":
        dtype = args.dtype if args.dtype is not None else "float16"
        model = Mistral31DialectEvaluator(
            model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            dtype=args.dtype,
            device=args.device
        )
    elif args.model == "gemma3_vllm":
        model = Gemma3DialectEvaluator(
            model_id="google/gemma-3-27b-it",
            dtype=args.dtype,
            device=args.device
        )
    elif args.model == "phi4_vllm":
        model = Phi4DialectEvaluator(
            model_id="microsoft/phi-4",
            dtype=args.dtype,
            device=args.device
        )
    elif args.model == "deepseekr1_vllm":
        model = DeepseekR1DialectEvaluator(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            dtype=args.dtype,
            device=args.device
        )

    # Create evaluator
    evaluator = DialectEvaluator(
        model=model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    evaluator.evaluate(
        data_file=args.data_file,
        sae_context_col=args.sae_context,
        aae_context_col=args.aae_context,
        sae_cont_col=args.sae_continuation,
        aae_cont_col=args.aae_continuation,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    # Check for required packages
    required_packages = ["vllm", "sentence_transformers", "transformers", "torch", "sklearn", "numpy", "pandas", "matplotlib"]
    import importlib.util
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.warning("Installing missing packages...")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call(["pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except Exception as e:
                logger.error(f"Failed to install {package}: {str(e)}")
    
    main()
