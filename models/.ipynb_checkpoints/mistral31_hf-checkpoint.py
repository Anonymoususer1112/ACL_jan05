from __future__ import annotations


from cProfile import label
import os
import re
from urllib import response

from cv2 import log
from gguf import Literal
from regex import F
from sklearn.manifold import trustworthiness
from sympy import im
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel

from .model_interface import TransformersModelInterface

logger = logging.getLogger(__name__)

PairType = Union[Tuple[str, str], Dict[str, str]]  # (sae, aae) or {"sae":..., "aae":...}


class Mistral31HFModelInterface(TransformersModelInterface):

    def __init__(self, model_path: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"):
        super().__init__("mistral31_hf", model_path)

    def _initialize_model(self):
        try:
            logger.info(f"Loading Mistral-3.1-HF from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, fix_mistral_regex=True, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={i: "44GB" for i in range(torch.cuda.device_count())},
            )
            logger.info("Mistral-3.1-HF loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Mistral-3.1-HF: {e}")
            raise e
        

    def call_model(self, text: str, max_tokens: int = 512) -> str:
        try:
            # Format as instruction for better results
            messages = [
                {"role": "user", "content": text}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_text = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error Calling Mistral31_hf: {str(e)}")
            return f"ERROR: {str(e)}"
        
    def get_sentence_activation(self, text: str, hs_idx: int = -1, max_length: int = 128) -> torch.Tensor:
        """
        Returns a single sentence embedding from hidden_states[hs_idx] using masked mean pooling.
        """
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,  # single sentence -> no need to pad
        ).to(self.model.device)

        def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """
            hidden_states: [B, T, D]
            attention_mask: [B, T] with 1 for real tokens, 0 for padding
            returns: [B, D]
            """
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # [B, T, 1]
            summed = (hidden_states * mask).sum(dim=1)                  # [B, D]
            denom = mask.sum(dim=1).clamp(min=1.0)                      # [B, 1]
            return summed / denom

        with torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True)
            hs = out.hidden_states[hs_idx]  # [1, T, D]
            sent = masked_mean_pool(hs, enc["attention_mask"])[0]   # [D]
            return sent.detach().cpu()

    def make_all_hidden_state_indices(self) -> List[int]:
        """
        Return all valid hidden_state indices.
        Usually: 0 ... num_layers
        (0 = embedding output, num_layers = final layer output)
        """
        num_layers = self.get_num_layers()
        return list(range(1, num_layers + 1)) 

    def get_num_layers(self) -> int:
        """
        Robustly infer number of transformer layers for HF models,
        including multimodal wrappers (e.g., configs with text_config / language_config).
        """
        import torch.nn as nn

        cfg = getattr(self.model, "config", None)

        # ---- helpers ----
        KEY_CANDIDATES = {
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "n_layers",
            "decoder_layers",
            "num_transformer_layers",
        }

        def _extract_from_config_obj(c) -> int | None:
            """Try common attributes on a config-like object."""
            if c is None:
                return None
            for k in KEY_CANDIDATES:
                v = getattr(c, k, None)
                if isinstance(v, int) and v > 0:
                    return v

            # Common nested configs in multimodal / wrapper models
            for subk in ["text_config", "language_config", "llm_config", "decoder_config"]:
                sub = getattr(c, subk, None)
                n = _extract_from_config_obj(sub)
                if isinstance(n, int) and n > 0:
                    return n

            return None

        def _extract_from_config_dict(d) -> int | None:
            """Recursively search dict for layer-count keys."""
            if not isinstance(d, dict):
                return None
            for k, v in d.items():
                if k in KEY_CANDIDATES and isinstance(v, int) and v > 0:
                    return v
                if isinstance(v, dict):
                    n = _extract_from_config_dict(v)
                    if isinstance(n, int) and n > 0:
                        return n
            return None

        def _find_best_modulelist(m: nn.Module) -> int | None:
            """
            Find the most likely transformer block list by scanning ModuleLists.
            We pick the largest "reasonable" ModuleList length.
            """
            best = None
            best_name = None
            for name, module in m.named_modules():
                if isinstance(module, nn.ModuleList):
                    try:
                        L = len(module)
                    except Exception:
                        continue
                    # Heuristic: transformer layers are usually >= 8
                    if L >= 8:
                        if best is None or L > best:
                            best = L
                            best_name = name
            return best

        # ---- 1) config object direct + nested ----
        n = _extract_from_config_obj(cfg)
        if isinstance(n, int) and n > 0:
            return n

        # ---- 2) config dict recursive ----
        if cfg is not None:
            try:
                d = cfg.to_dict()
                n = _extract_from_config_dict(d)
                if isinstance(n, int) and n > 0:
                    return n
            except Exception:
                pass

        # ---- 3) module tree fallback: scan ModuleLists ----
        try:
            n = _find_best_modulelist(self.model)
            if isinstance(n, int) and n > 0:
                return n
        except Exception:
            pass

        # ---- fail with richer debug ----
        cfg_keys = []
        if cfg is not None:
            try:
                cfg_keys = sorted(list(cfg.to_dict().keys()))
            except Exception:
                cfg_keys = sorted([k for k in dir(cfg) if not k.startswith("_")])

        raise ValueError(
            "Cannot infer number of layers for this model. "
            "Tried nested config search + config.to_dict recursive search + ModuleList scan. "
            f"Top-level config keys (sample): {cfg_keys[:80]}"
        )


    
    def compute_aae_dialect_vector_for_layer(
        self,
        pairs: List[PairType],
        hs_idx: int,
        max_pairs: int | None = None,
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        sae_acts, aae_acts = [], []
        def _unpack_pair(p: PairType) -> Tuple[str, str]:
            if isinstance(p, dict):
                return p["sae"], p["aae"]
            return p[0], p[1]

        for p in tqdm(pairs, desc=f"Computing activations for hidden_states[{hs_idx}]"):
            sae_text, aae_text = _unpack_pair(p)
            a_vec = self.get_sentence_activation(aae_text, hs_idx=hs_idx, max_length=max_length)
            s_vec = self.get_sentence_activation(sae_text, hs_idx=hs_idx, max_length=max_length)
            aae_acts.append(a_vec)
            sae_acts.append(s_vec)

        sae_acts = torch.stack(sae_acts)  # [N, D]
        aae_acts = torch.stack(aae_acts)  # [N, D]

        v_aae = (aae_acts - sae_acts).mean(dim=0)                   # [D]
        v_aae_unit = v_aae / (v_aae.norm() + 1e-8)

        proj_sae = sae_acts @ v_aae_unit                            # [N]
        proj_aae = aae_acts @ v_aae_unit                            # [N]

        stats = {
            "proj_mean_SAE": proj_sae.mean().item(),
            "proj_std_SAE": proj_sae.std(unbiased=False).item(),
            "proj_mean_AAE": proj_aae.mean().item(),
            "proj_std_AAE": proj_aae.std(unbiased=False).item(),
            "mean_gap_AAE_minus_SAE": (proj_aae - proj_sae).mean().item(),
        }

        return v_aae_unit, stats 
    

    def layer_sweep_dialect_vectors_full(
        self,
        pairs: List[Tuple[str, str]],
        max_pairs: int = 200,
        max_length: int = 128,
    ) -> Tuple[Dict[int, Dict[str, Any]], int]:

        num_layers = self.get_num_layers()
        logger.info(f"Model has {num_layers} transformer layers.")

        candidate_layers = list(range(1, num_layers + 1))
        logger.info(f"Sweeping all hidden_state indices: {candidate_layers}")

        results = {}

        for hs_idx in tqdm(candidate_layers, desc="Full layer sweep (dialect vectors)"):
            v_aae, stats = self.compute_aae_dialect_vector_for_layer(
                pairs=pairs,
                hs_idx=hs_idx,
                max_pairs=max_pairs,
                max_length=max_length
            )
            results[hs_idx] = {
                "v_aae_unit": v_aae,
                "stats": stats
            }

        best_layer = max(
            results,
            key=lambda i: results[i]["stats"]["mean_gap_AAE_minus_SAE"]
        )

        logger.info(
            f"Best hidden_state index: {best_layer} "
            f"(gap={results[best_layer]['stats']['mean_gap_AAE_minus_SAE']:.4f})"
        )

        return results, best_layer
    
def call_mistral31_hf(text: str) -> str:
    model_interface = Mistral31HFModelInterface()
    return model_interface.call_model(text)