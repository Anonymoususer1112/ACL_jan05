from __future__ import annotations

from contextlib import contextmanager, nullcontext
from cProfile import label
import os
import re
from urllib import response

from cv2 import log
from gguf import Literal
from regex import F
from sympy import im
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .model_interface import TransformersModelInterface

logger = logging.getLogger(__name__)

PairType = Union[Tuple[str, str], Dict[str, str]]  # (sae, aae) or {"sae":..., "aae":...}


class Phi4HFModelInterface(TransformersModelInterface):

    def __init__(self, model_path: str = "microsoft/phi-4"):
        super().__init__("phi4_hf", model_path)

    def _initialize_model(self):
        try:
            logger.info(f"Loading Phi-4-HF from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={i: "44GB" for i in range(torch.cuda.device_count())},
            )
            logger.info("Phi-4-HF loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Phi-4-HF: {e}")
            raise e
        

    def call_model(self, text: str, max_tokens: int = 512) -> str:
        try:
            messages = [{"role": "user", "content": text}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt")

            # move inputs to embedding device (accelerate-safe)
            try:
                emb = self.model.get_input_embeddings()
                target_device = next(emb.parameters()).device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            except Exception:
                pass

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error Calling Phi4_hf: {str(e)}")
            return f"ERROR: {str(e)}"

            
        except Exception as e:
            logger.error(f"Error Calling Phi4_hf: {str(e)}")
            return f"ERROR: {str(e)}"
        
    def get_sentence_activation(self, text: str, hs_idx: int = -1, max_length: int = 128) -> torch.Tensor:
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        # move to embedding device (accelerate-safe)
        try:
            emb = self.model.get_input_embeddings()
            target_device = next(emb.parameters()).device
            enc = {k: v.to(target_device) for k, v in enc.items()}
        except Exception:
            pass

        def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom

        with torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True)
            hs = out.hidden_states[hs_idx]  # [1, T, D]
            sent = masked_mean_pool(hs, enc["attention_mask"])[0]
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
    
    # -------------------------
    # 1) locate transformer blocks robustly
    # -------------------------
    def _get_transformer_blocks(self):
        # --- cache ---
        if hasattr(self, "_blocks_cache") and self._blocks_cache is not None:
            return self._blocks_cache

        m = self.model
        candidates = [
            ("model.layers", lambda x: x.model.layers),
            ("model.model.layers", lambda x: x.model.model.layers),
            ("transformer.h", lambda x: x.transformer.h),
            ("gpt_neox.layers", lambda x: x.gpt_neox.layers),
            ("backbone.layers", lambda x: x.backbone.layers),
        ]

        for name, getter in candidates:
            try:
                blocks = getter(m)
                if blocks is not None and len(blocks) > 0:
                    self._blocks_cache = blocks
                    if not getattr(self, "_blocks_logged", False):
                        logger.info(f"[phi4 hf] Using blocks from {name}, n={len(blocks)}")
                        self._blocks_logged = True
                    return blocks
            except Exception:
                pass

        for attr_name in ["layers", "h", "blocks"]:
            if hasattr(m, attr_name):
                blocks = getattr(m, attr_name)
                try:
                    if blocks is not None and len(blocks) > 0:
                        self._blocks_cache = blocks
                        if not getattr(self, "_blocks_logged", False):
                            logger.info(f"[phi4 hf] Using blocks from model.{attr_name}, n={len(blocks)}")
                            self._blocks_logged = True
                        return blocks
                except Exception:
                    pass

        raise ValueError("Cannot find transformer blocks for phi4_hf.")



    # -------------------------
    # 2) steering hook (device_map safe)
    # -------------------------
    @contextmanager
    def steering_hook(self, layer_id: int, v_aae_unit: torch.Tensor, beta: float):
        """
        Apply activation steering at a given transformer block output:
        hidden_states += beta * v
        layer_id convention: 1..num_layers (same as your sweep hidden_state index)
        """
        blocks = self._get_transformer_blocks()
        block_idx = int(layer_id) - 1
        if block_idx < 0 or block_idx >= len(blocks):
            raise ValueError(f"layer_id={layer_id} out of range for blocks len={len(blocks)}")

        block = blocks[block_idx]

        # cache v per device/dtype to avoid repeated .to()
        v_cache = {}

        def _get_v_like(hidden: torch.Tensor):
            key = (hidden.device, hidden.dtype)
            if key in v_cache:
                return v_cache[key]
            v_dev = v_aae_unit.to(device=hidden.device, dtype=hidden.dtype)
            v_cache[key] = v_dev
            return v_dev

        def _hook(module, inputs, output):
            # output could be Tensor or tuple(Tensor, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                v_like = _get_v_like(hidden)
                steered = hidden + (beta * v_like).view(1, 1, -1)
                return (steered,) + output[1:]
            else:
                hidden = output
                v_like = _get_v_like(hidden)
                return hidden + (beta * v_like).view(1, 1, -1)

        handle = block.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()


    # -------------------------
    # 3) steered call_model (generation)
    # -------------------------
    def call_model_steered(
        self,
        text: str,
        layer_id: int,
        v_aae_unit: torch.Tensor,
        beta: float,
        max_tokens: int = 512,
        do_sample: bool = False,   # deterministic by default for eval
        temperature: float = 0.0,  # ignored if do_sample=False
        top_p: float = 1.0,
    ) -> str:
        """Call Phi-4-HF with optional activation steering (inference-time)."""
        try:
            messages = [{"role": "user", "content": text}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # IMPORTANT: with device_map="auto", do NOT force inputs to self.model.device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Let HF dispatch handle device placement, but move input_ids to first device if needed.
            # A safe approach is to move to the model's input embedding device:
            try:
                emb = self.model.get_input_embeddings()
                target_device = next(emb.parameters()).device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            except Exception:
                # fallback: leave on CPU (HF may still move internally)
                pass

            ctx = self.steering_hook(layer_id=layer_id, v_aae_unit=v_aae_unit, beta=beta) if beta != 0.0 else nullcontext()

            with ctx:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 0.0,
                        top_p=top_p if do_sample else 1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,  # consistent with your phi3 setting
                    )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error calling Phi4_hf (steered): {str(e)}")
            return f"ERROR: {str(e)}"


    # -------------------------
    # 4) steered continuation preference (A/B)
    # -------------------------
    def get_continuation_preference_steered(
        self,
        context_sae: str,
        continuation_sae: str,
        context_aae: str,
        continuation_aae: str,
        context_variant: Literal["SAE", "AAE"] = "SAE",
        rng: Optional[random.Random] = None,
        prompt_setting: str = "sae",
        *,
        layer_id: int,
        v_aae_unit: torch.Tensor,
        beta: float = 0.0,
        max_tokens: int = 8,
        do_sample: bool = False,
    ) -> Dict[str, Any]:
        """
        Same protocol as get_continuation_preference, but evaluated under steering.
        """
        r = rng if rng is not None else random

        if str(context_variant).upper() == "AAE":
            context_text = context_aae
            context_variant_norm = "AAE"
        else:
            context_text = context_sae
            context_variant_norm = "SAE"

        swap = r.random() < 0.5
        if swap:
            A_cont, B_cont = continuation_aae, continuation_sae
            label_map = {"A": "AAE", "B": "SAE"}
        else:
            A_cont, B_cont = continuation_sae, continuation_aae
            label_map = {"A": "SAE", "B": "AAE"}

        if prompt_setting.lower() == "sae":
            prompt = (
                'Must pick one continuation that best follows the context.\n'
                'Answer only "A" or "B".\n\n'
                f'Context: "{context_text}"\n'
                f'A: "{A_cont}"\n'
                f'B: "{B_cont}"\n'
            )
        else:
            raise ValueError(f"Unknown prompt_setting: {prompt_setting}")

        max_retries = 3
        last_raw_response = ""

        for attempt in range(1, max_retries + 1):
            prompt_try = prompt if attempt == 1 else prompt + '\nIMPORTANT: Output exactly ONE character: "A" or "B".\n'

            raw = self.call_model_steered(
                text=prompt_try,
                layer_id=layer_id,
                v_aae_unit=v_aae_unit,
                beta=beta,
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=0.0,
                top_p=1.0,
            )
            last_raw_response = raw
            resp = raw.strip().upper()

            if resp == "A":
                choice = "A"
            elif resp == "B":
                choice = "B"
            else:
                m = re.search(r"\b([AB])\b", resp)
                choice = m.group(1) if m else "ERROR"

            if choice in label_map:
                return {
                    "preferred": label_map[choice],
                    "A_is": label_map["A"],
                    "B_is": label_map["B"],
                    "context_variant": context_variant_norm,
                    "beta": float(beta),
                    "layer_id": int(layer_id),
                    "raw_response": raw,
                }

        return {
            "preferred": "ERROR",
            "A_is": label_map["A"],
            "B_is": label_map["B"],
            "context_variant": context_variant_norm,
            "beta": float(beta),
            "layer_id": int(layer_id),
            "raw_response": last_raw_response,
        }


    # -------------------------
    # 5) log-prob utilities (global logP version)
    # -------------------------
    def _get_log_prob_for_text(self, text: str) -> float:
        """
        Log P(text) using teacher forcing:
        sum_t log p(x_t | x_<t)
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)

        # place inputs on embedding device (device_map safe)
        try:
            emb = self.model.get_input_embeddings()
            target_device = next(emb.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        except Exception:
            pass

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        if input_ids.shape[1] < 2:
            return -float("inf")

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B,T,V]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target = input_ids[:, 1:].unsqueeze(2)
        token_lp = torch.gather(log_probs[:, :-1, :], 2, target).squeeze(2)
        mask = attention_mask[:, 1:]
        return float((token_lp * mask).sum().item())


    def _get_log_prob_for_text_steered(self, text: str, layer_id: int, v_aae_unit: torch.Tensor, beta: float) -> float:
        """
        Same as _get_log_prob_for_text but under steering_hook.
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        try:
            emb = self.model.get_input_embeddings()
            target_device = next(emb.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        except Exception:
            pass

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        if input_ids.shape[1] < 2:
            return -float("inf")

        ctx = self.steering_hook(layer_id=layer_id, v_aae_unit=v_aae_unit, beta=beta) if beta != 0.0 else nullcontext()

        with ctx:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target = input_ids[:, 1:].unsqueeze(2)
        token_lp = torch.gather(log_probs[:, :-1, :], 2, target).squeeze(2)
        mask = attention_mask[:, 1:]
        return float((token_lp * mask).sum().item())


    def calculate_log_difference_for_preference_steered(
        self,
        context_sae: str,
        continuation_sae: str,
        context_aae: str,
        continuation_aae: str,
        context_variant: str = "SAE",
        *,
        layer_id: int,
        v_aae_unit: torch.Tensor,
        beta: float,
    ) -> Tuple[float, float, float]:
        """
        Returns: (logp_aae, logp_sae, logdiff) where
        logdiff = logp_sae - logp_aae
        """
        if str(context_variant).upper() == "AAE":
            context_used = str(context_aae)
        else:
            context_used = str(context_sae)

        sae_full = context_used + " " + str(continuation_sae)
        aae_full = context_used + " " + str(continuation_aae)

        logp_sae = self._get_log_prob_for_text_steered(sae_full, layer_id=layer_id, v_aae_unit=v_aae_unit, beta=beta)
        logp_aae = self._get_log_prob_for_text_steered(aae_full, layer_id=layer_id, v_aae_unit=v_aae_unit, beta=beta)
        return logp_aae, logp_sae, float(logp_sae - logp_aae)


    # -------------------------
    # 6) activation-level shift helpers
    # -------------------------
    def get_sentence_activation_steered(self, text: str, hs_idx: int, layer_id: int, v_aae_unit: torch.Tensor, beta: float, max_length: int = 128) -> torch.Tensor:
        """
        Return pooled sentence embedding from hidden_states[hs_idx] under steering.
        """
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        # move to embedding device (device_map safe)
        try:
            emb = self.model.get_input_embeddings()
            target_device = next(emb.parameters()).device
            enc = {k: v.to(target_device) for k, v in enc.items()}
        except Exception:
            pass

        def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom

        ctx = self.steering_hook(layer_id=layer_id, v_aae_unit=v_aae_unit, beta=beta) if beta != 0.0 else nullcontext()
        with ctx:
            with torch.inference_mode():
                out = self.model(**enc, output_hidden_states=True)
                hs = out.hidden_states[hs_idx]
                sent = masked_mean_pool(hs, enc["attention_mask"])[0]
                return sent.detach().cpu()


    def get_hook_projection(self, text: str, layer_id: int, v_aae_unit: torch.Tensor, beta: float, max_length: int = 128, hs_idx: Optional[int] = None) -> float:
        """
        Convenience: run get_sentence_activation_steered then project onto v_aae_unit.
        If hs_idx is None, default to layer_id (same as your phi3 usage).
        """
        if hs_idx is None:
            hs_idx = int(layer_id)
        sent = self.get_sentence_activation_steered(
            text=text,
            hs_idx=int(hs_idx),
            layer_id=int(layer_id),
            v_aae_unit=v_aae_unit,
            beta=float(beta),
            max_length=max_length,
        )
        v = v_aae_unit.detach().float().cpu()
        v = v / (v.norm() + 1e-8)
        return float((sent.float() @ v).item())
    
    def get_sentiment_steered(
        self,
        text: str,
        *,
        layer_id: int,
        v_aae_unit: torch.Tensor,
        beta: float,
        max_tokens: int = 8,
        do_sample: bool = False,
    ) -> Dict[str, Any]:
        """
        Steered sentiment classification for Phi-4-HF.
        Output must be one of: positive / negative / neutral
        """

        prompt = f"""
    You are a sentiment classifier. Output EXACTLY one label from this set:
    positive, negative, neutral
    Do NOT output anything else.

    Text: "{text}"

    Label:
    """.strip()

        try:
            resp = self.call_model_steered(
                text=prompt,
                layer_id=layer_id,
                v_aae_unit=v_aae_unit,
                beta=float(beta),
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=0.0,
                top_p=1.0,
            )
            raw = (resp or "").strip()
            s = raw.lower().strip()

            # strict match first
            if s == "positive":
                sentiment, score = "positive", 1
            elif s == "negative":
                sentiment, score = "negative", -1
            elif s == "neutral":
                sentiment, score = "neutral", 0
            else:
                # fallback contains (robust to "Label: positive")
                if ("positive" in s) and ("negative" not in s):
                    sentiment, score = "positive", 1
                elif "negative" in s:
                    sentiment, score = "negative", -1
                elif "neutral" in s:
                    sentiment, score = "neutral", 0
                else:
                    sentiment, score = "ERROR", 0

            return {"sentiment": sentiment, "score": score, "raw_response": raw}

        except Exception as e:
            logger.error(f"Error in get_sentiment_steered: {str(e)}")
            return {"sentiment": "ERROR", "score": 0, "raw_response": str(e)}

    
def call_phi4_hf(text: str) -> str:
    model_interface = Phi4HFModelInterface()
    return model_interface.call_model(text)