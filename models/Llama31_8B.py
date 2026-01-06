from __future__ import annotations


from cProfile import label
import os
import re
import tokenize
from urllib import response

from cv2 import log
from gguf import Literal
from regex import F
from sympy import im
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .model_interface import TransformersModelInterface

logger = logging.getLogger(__name__)

class Llama318BInterface(TransformersModelInterface):
    """Interface for Llama 3.1 8B model using Hugging Face transformers."""
    
    def __init__(self, model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__("llama31_8b", model_path)

    def _initialize_model(self):
        try:
            logger.info(f"Loading Llama 3.1 8B from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=False,
            ).to(self.device)
            logger.info("Llama 3.1 8B model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Llama 3.1 8B model: {str(e)}")
            raise
        
    def call_model(self, text: str, max_tokens: int = 512) -> str:
        """Function to call Llama 3.1 8B model.
        
        Args:
            text: Input text/prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response
        """
        try:
            messages = [
                {"role": "user", "content": text}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Extract generated text
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during Llama 3.1 8B model call: {str(e)}")
            return "ERROR: Model call failed"
        
    def get_sentence_activation(self, text: str, layer_idx: int = -1, max_length: int = 128) -> Any:
        """Get sentence activation from a specific layer.
        
        Args:
            text: Input text
            layer_idx: Layer index to extract activations from (default: last layer)
        Returns:
            Numpy array of activations
        """
        input = self.tokenizer(text, return_tensors="pt", max_length=max_length).to(self.model.device)
        with torch.no_grad():
            out = self.model(**input, output_hidden_states=True)

            # hidden_states is a tuple of (layer0, layer1, ..., layerN)
            hidden_states = out.hidden_states[layer_idx]
            # Get the activations for the last token or mean over tokens
            sent_act = hidden_states.mean(dim=1).squeeze(0).cpu()

        return sent_act

    def get_num_layers(self) -> int:
        """Get the number of layers in the model.
        
        Returns:
            Number of transformer layers
        """
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        
        # Some models may have different naming mechanisms
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            return len(self.model.transformer.layers)
        raise ValueError("Cannot infer number of layers for this model.")
    
    def compute_aae_dialect_vector_for_layer(
            self,
            pairs: List[Dict[str, str]], # List of dicts with 'sae' and 'aae' keys
            layer_idx: int,
            max_pairs: int | None = None,
            max_length: int = 128
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        """Compute AAE dialect vector for a specific layer.

        Args:
            pairs: List of dicts with 'sae' and 'aae' keys containing sentence
            layer_idx: Layer index to extract activations from
            max_pairs: Maximum number of pairs to use (for debugging)
            max_length: Maximum token length for each sentence
        Returns:
            Tuple of:
                - AAE dialect vector (torch.Tensor)
                - Statistics dictionary with projection means and stds
        """
        
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        sae_acts = []
        aae_acts = []

        for sae_text, aae_text in tqdm(pairs, desc=f"Computing activations for layer {layer_idx}"):
            a_vec = self.get_sentence_activation(aae_text, layer_idx=layer_idx, max_length=max_length)
            s_vec = self.get_sentence_activation(sae_text, layer_idx=layer_idx, max_length=max_length)

            aae_acts.append(a_vec)
            sae_acts.append(s_vec)

        sae_acts = torch.stack(sae_acts)
        aae_acts = torch.stack(aae_acts)

        v_aae = (aae_acts - sae_acts).mean(dim=0)
        v_aae_unit = v_aae / (v_aae.norm() + 1e-8)

        proj_sae = (sae_acts @ v_aae_unit)
        proj_aae = (aae_acts @ v_aae_unit)

        stats = {
            "proj_mean_SAE": proj_sae.mean().item(),
            "proj_std_SAE": proj_sae.std().item(),
            "proj_mean_AAE": proj_aae.mean().item(),
            "proj_std_AAE": proj_aae.std().item(),
            "mean_gap_AAE_minus_SAE": (proj_aae - proj_sae).mean().item(),
        }

        return v_aae, stats
    
    def make_candidate_layers(
            self,
            fractions: Tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    ) -> List[int]:
        """Get candidate layers for dialect vector computation.
        Args:
            fractions: Tuple of fractions to determine layer indices
        Returns:
            List of layer indices
        """
        num_layers = self.get_num_layers()
        layers = []
        for f in fractions:
            lid = int(round(num_layers * f))
            lid = min(max(lid, 0), num_layers)
            layers.append(lid)
        
        layers = sorted(set(layers))
        return layers
    
    def layer_sweep_dialect_vectors_auto(
            self,
            pairs: List[Tuple[str, str]],
            fractions: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
            max_pairs: int = 200,
            max_length: int = 128,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor | Dict]], int]:

        num_layers = self.get_num_layers()
        logger.info(f"Model has {num_layers} layers. Computing dialect vectors at candidate layers.")

        candidate_layers = self.make_candidate_layers(fractions=fractions)
        logger.info(f"Candidate layers for dialect vector computation: {candidate_layers}")

        results: Dict[int, Dict[str, torch.Tensor | Dict]] = {}

        for layer_idx in tqdm(candidate_layers, desc="Computing dialect vectors for layers"):
            logger.info(f"Computing AAE dialect vector for layer {layer_idx}...")
            v_aae, stats = self.compute_aae_dialect_vector_for_layer(
                pairs=pairs,
                layer_idx=layer_idx,
                max_pairs=max_pairs,
                max_length=max_length
            )
            logger.info(
                f"Layer {layer_idx}: mean projection SAE={stats['proj_mean_SAE']:.4f}, "
                f"AAE={stats['proj_mean_AAE']:.4f}, gap={stats['mean_gap_AAE_minus_SAE']:.4f}"
            )
            results[layer_idx] = {
                "v_aae": v_aae,
                "stats": stats
            }
        
        best_layer = max(
            results.keys(),
            key=lambda lid: results[lid]["stats"]["mean_gap_AAE_minus_SAE"]
        )
        logger.info(f"Best layer for AAE dialect vector: {best_layer} with gap {results[best_layer]['stats']['mean_gap_AAE_minus_SAE']:.4f}")
        return results, best_layer
        

def call_llama31_8b_model(text: str) -> str:
    """Utility function to call Llama 3.1 8B model with a prompt.
    Args:
        prompt: Input text/prompt
    Returns:
        Model response
    """

    model = Llama318BInterface()
    return model.call_model(text)