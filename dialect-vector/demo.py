#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conditional PPL with optional inference-time forward hook (dialect steering).

Compute PPL(continuation | context) for 4 combinations:
  SAE_ctx + SAE_cont
  SAE_ctx + AAE_cont
  AAE_ctx + SAE_cont
  AAE_ctx + AAE_cont

Adds:
  - --vector_pt: layer vector pt (supports key v_aae or v_aae_unit)
  - --hs_idx: hidden_states index (default 38)
  - --beta: steering strength (default 0.0)
  - --beta_grid: comma-separated list of betas to sweep; if provided, runs all betas
  - always use_cache=False during forward (stability)

Outputs (per beta):
  - ppl_scores_long.csv
  - ppl_summary_by_combo.csv
  - ppl_top_outliers.csv
  - ppl_failed_rows.csv
"""

import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager, nullcontext

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------
# Hook utilities
# -------------------------
def get_decoder_blocks(model):
    # Phi / Llama style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # NeoX style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Cannot find transformer blocks for this model.")

def hs_idx_to_block_idx(model, hs_idx: int) -> int:
    blocks = get_decoder_blocks(model)
    block_idx = hs_idx - 1  # hs[0]=emb, hs[1]=after block0
    if not (0 <= block_idx < len(blocks)):
        raise ValueError(f"hs_idx={hs_idx} -> block_idx={block_idx} out of range (num_blocks={len(blocks)})")
    return block_idx

def load_v_unit(pt_path: str) -> torch.Tensor:
    """
    Accepts:
      - {"v_aae_unit": tensor} or {"v_aae": tensor} or direct tensor
    Returns unit-norm tensor on CPU float32.
    """
    obj = torch.load(pt_path, map_location="cpu")
    if torch.is_tensor(obj):
        v = obj
    elif isinstance(obj, dict):
        if "v_aae_unit" in obj and torch.is_tensor(obj["v_aae_unit"]):
            v = obj["v_aae_unit"]
        elif "v_aae" in obj and torch.is_tensor(obj["v_aae"]):
            v = obj["v_aae"]
        else:
            # fallback: first tensor field
            v = None
            for _, val in obj.items():
                if torch.is_tensor(val):
                    v = val
                    break
            if v is None:
                raise KeyError(f"No tensor found in {pt_path}. keys={list(obj.keys())}")
    else:
        raise TypeError(f"Unsupported pt content type: {type(obj)}")

    v = v.float()
    v = v / (v.norm() + 1e-8)
    return v

@contextmanager
def dialect_steering_hook(model, v_unit_cpu: torch.Tensor, hs_idx: int, beta: float):
    """
    Adds: hidden <- hidden + beta * v_unit
    at the output of transformer block (hs_idx-1).

    v_unit_cpu is kept on CPU; inside hook we move it to output's device/dtype.
    """
    if beta == 0.0:
        yield
        return

    blocks = get_decoder_blocks(model)
    block_idx = hs_idx_to_block_idx(model, hs_idx)

    printed = {"done": False}

    def hook_fn(module, inputs, output):
        def add_vec(x: torch.Tensor) -> torch.Tensor:
            v = v_unit_cpu.to(device=x.device, dtype=x.dtype, non_blocking=True)
            y = x + beta * v
            # one-time tiny debug (optional)
            if not printed["done"]:
                delta = (y - x).abs().mean().item()
                #logger.info(f"[HOOK] fired at block_idx={block_idx} beta={beta} mean|Δ|={delta:.6f} dev={x.device} dtype={x.dtype}")
                printed["done"] = True
            return y

        if torch.is_tensor(output):
            return add_vec(output)

        if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            out0 = add_vec(output[0])
            if isinstance(output, tuple):
                return (out0,) + tuple(output[1:])
            out_list = list(output)
            out_list[0] = out0
            return out_list

        return output

    handle = blocks[block_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def dialect_steering_hook_rms(
    model,
    v_unit_cpu: torch.Tensor,
    hs_idx: int,
    alpha: float,
    eps: float = 1e-6,
):
    """
    RMS-scaled steering:
      x <- x + alpha * RMS(x) * v_unit

    RMS(x) = sqrt(mean(x^2, dim=-1, keepdim=True))

    v_unit_cpu stays on CPU; moved to x.device/dtype inside hook.
    """
    if alpha == 0.0:
        yield
        return

    blocks = get_decoder_blocks(model)
    block_idx = hs_idx_to_block_idx(model, hs_idx)

    printed = {"done": False}

    def hook_fn(module, inputs, output):
        def add_rms_scaled(x: torch.Tensor) -> torch.Tensor:
            v = v_unit_cpu.to(device=x.device, dtype=x.dtype, non_blocking=True)  # [D]
            rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)      # [B,T,1]
            y = x + alpha * rms * v                                              # broadcast v -> [B,T,D]

            if not printed["done"]:
                mean_delta = (y - x).abs().mean().item()
                mean_rms = rms.mean().item()
                logger.info(
                    f"[HOOK-RMS] fired block_idx={block_idx} hs_idx={hs_idx} alpha={alpha} "
                    f"mean_rms={mean_rms:.6f} mean|Δ|={mean_delta:.6f} dev={x.device} dtype={x.dtype}"
                )
                printed["done"] = True
            return y

        if torch.is_tensor(output):
            return add_rms_scaled(output)

        if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            out0 = add_rms_scaled(output[0])
            if isinstance(output, tuple):
                return (out0,) + tuple(output[1:])
            out_list = list(output)
            out_list[0] = out0
            return out_list

        return output

    handle = blocks[block_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
# -------------------------
# Core: conditional PPL
# -------------------------
@torch.inference_mode()
def conditional_ppl_batch(
    model,
    tokenizer,
    contexts: List[str],
    continuations: List[str],
    device: str,
    batch_size: int = 8,
    separator: str = "\n",
    max_length: Optional[int] = None,
    # ---- hook args ----
    v_unit_cpu: Optional[torch.Tensor] = None,
    hs_idx: int = 38,
    alpha: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Compute conditional perplexity: PPL(continuation | context)

    Returns list of dicts with:
      - n_ctx_tokens
      - n_cont_tokens
      - avg_nll_cont
      - log_ppl_cont
      - ppl_cont
    """

    assert len(contexts) == len(continuations)
    results: List[Dict[str, float]] = []

    # --------------------------------------------------
    # Build full texts and track token boundaries
    # --------------------------------------------------
    full_texts = []
    n_ctx_tokens_list = []
    n_full_tokens_list = []

    for ctx_raw, cont_raw in zip(contexts, continuations):
        ctx = str(ctx_raw).rstrip()
        cont = str(cont_raw).lstrip()

        ctx_boundary = ctx + separator
        full_text = ctx_boundary + cont

        ctx_ids = tokenizer.encode(ctx_boundary, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        full_texts.append(full_text)
        n_ctx_tokens_list.append(len(ctx_ids))
        n_full_tokens_list.append(len(full_ids))

    # --------------------------------------------------
    # Batch processing
    # --------------------------------------------------
    for i in tqdm(
        range(0, len(full_texts), batch_size),
        desc="Calculating PPL(cont|ctx)",
        dynamic_ncols=True,
    ):
        batch_texts = full_texts[i:i + batch_size]
        batch_ctx_lens = n_ctx_tokens_list[i:i + batch_size]
        batch_full_lens = n_full_tokens_list[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=(max_length is not None),
            max_length=max_length,
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # --------------------------------------------------
        # Create hook context PER BATCH (critical fix)
        # --------------------------------------------------
        ctx = (
            dialect_steering_hook_rms(
                model,
                v_unit_cpu=v_unit_cpu,
                hs_idx=hs_idx,
                alpha=alpha,
            )
            if (v_unit_cpu is not None and alpha != 0.0)
            else nullcontext()
        )

        with ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,   # IMPORTANT for Phi-3
            )
            logits = outputs.logits  # [B, T, V]

        # --------------------------------------------------
        # Log-prob computation
        # --------------------------------------------------
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, 1:]                      # [B, T-1]
        lp_next = torch.gather(
            log_probs[:, :-1, :],
            2,
            target_ids.unsqueeze(-1)
        ).squeeze(-1)                                      # [B, T-1]

        target_mask = attention_mask[:, 1:]                # [B, T-1]

        # --------------------------------------------------
        # Per-sample aggregation
        # --------------------------------------------------
        for b in range(input_ids.size(0)):
            ctx_len = int(batch_ctx_lens[b])
            full_len = int(batch_full_lens[b])

            eff_len = int(attention_mask[b].sum().item())
            eff_lp_len = max(eff_len - 1, 0)

            cont_start_lp = max(ctx_len - 1, 0)
            cont_end_lp = max(full_len - 1, 0)

            cont_start_lp = min(cont_start_lp, eff_lp_len)
            cont_end_lp = min(cont_end_lp, eff_lp_len)

            if cont_end_lp <= cont_start_lp:
                results.append({
                    "n_ctx_tokens": float(ctx_len),
                    "n_cont_tokens": 0.0,
                    "avg_nll_cont": float("nan"),
                    "log_ppl_cont": float("nan"),
                    "ppl_cont": float("nan"),
                })
                continue

            cont_lp = lp_next[b, cont_start_lp:cont_end_lp]
            cont_m = target_mask[b, cont_start_lp:cont_end_lp]

            cont_lp = cont_lp[cont_m.bool()]
            n_cont = int(cont_lp.numel())

            if n_cont == 0:
                results.append({
                    "n_ctx_tokens": float(ctx_len),
                    "n_cont_tokens": 0.0,
                    "avg_nll_cont": float("nan"),
                    "log_ppl_cont": float("nan"),
                    "ppl_cont": float("nan"),
                })
                continue

            avg_nll = float((-cont_lp.sum() / n_cont).item())
            ppl = float(math.exp(avg_nll))

            results.append({
                "n_ctx_tokens": float(ctx_len),
                "n_cont_tokens": float(n_cont),
                "avg_nll_cont": avg_nll,
                "log_ppl_cont": avg_nll,
                "ppl_cont": ppl,
            })

    return results


# -------------------------
# Main evaluator helpers
# -------------------------
def build_long_df(
    df: pd.DataFrame,
    sae_ctx_col: str,
    aae_ctx_col: str,
    sae_cont_col: str,
    aae_cont_col: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    sae_ctx = df[sae_ctx_col].astype(str).tolist()
    aae_ctx = df[aae_ctx_col].astype(str).tolist()
    sae_cont = df[sae_cont_col].astype(str).tolist()
    aae_cont = df[aae_cont_col].astype(str).tolist()

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
    return long_df, contexts_for_scoring, conts_for_scoring


def save_outputs(long_df: pd.DataFrame, out_dir: Path):
    long_path = out_dir / "ppl_scores_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {long_path}")

    failed_df = long_df[long_df["n_cont_tokens"] <= 0].copy()
    failed_path = out_dir / "ppl_failed_rows.csv"
    failed_df.to_csv(failed_path, index=False, encoding="utf-8")
    logger.info(f"Saved failed rows: {failed_path} (count={len(failed_df)})")

    summary = (long_df
        .groupby("combo")
        .agg(
            n_total=("ppl_cont", "size"),
            n_valid=("ppl_cont", lambda x: int(np.isfinite(x).sum())),
            mean_log_ppl=("log_ppl_cont", "mean"),
            std_log_ppl=("log_ppl_cont", "std"),
            median_ppl=("ppl_cont", "median"),
            p95_ppl=("ppl_cont", lambda x: float(np.nanpercentile(x, 95))),
            mean_n_ctx_tokens=("n_ctx_tokens", "mean"),
            mean_n_cont_tokens=("n_cont_tokens", "mean"),
        )
        .reset_index()
        .sort_values("combo")
    )

    if (summary["combo"] == "SAE_SAE").any():
        baseline = float(summary.loc[summary["combo"] == "SAE_SAE", "mean_log_ppl"].values[0])
        summary["delta_log_ppl_vs_SAE_SAE"] = summary["mean_log_ppl"] - baseline
        summary["ppl_ratio_vs_SAE_SAE"] = np.exp(summary["delta_log_ppl_vs_SAE_SAE"])

    summary_path = out_dir / "ppl_summary_by_combo.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {summary_path}")

    top_outliers = (long_df
        .sort_values("ppl_cont", ascending=False)
        .groupby("combo")
        .head(20)
    )
    outlier_path = out_dir / "ppl_top_outliers.csv"
    top_outliers.to_csv(outlier_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {outlier_path}")


# -------------------------
# Main
# -------------------------
def parse_alpha_grid(s: Optional[str]) -> Optional[List[float]]:
    if s is None or str(s).strip() == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--sae_context", type=str, default="sae_context")
    parser.add_argument("--aae_context", type=str, default="aae_context")
    parser.add_argument("--sae_continuation", type=str, default="sae_continuation")
    parser.add_argument("--aae_continuation", type=str, default="aae_continuation")

    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-medium-4k-instruct")
    parser.add_argument("--output_dir", type=str, default="output_evaluations/phi3_medium_ppl")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--separator", type=str, default="\n")
    parser.add_argument("--max_length", type=int, default=None, help="Optional truncation length in tokens")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Hook args
    parser.add_argument("--vector_pt", type=str, default=None, help="Path to layer vector pt (v_aae or v_aae_unit)")
    parser.add_argument("--hs_idx", type=int, default=38, help="hidden_states index to hook (default 38)")
    parser.add_argument("--alpha", type=float, default=0.0, help="RMS-scaled steering strength (default 0)")
    parser.add_argument("--alpha_grid", type=str, default=None, help="comma-separated alphas; if set, sweep all and ignore --alpha")


    # Phi-3 remote code
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for Phi-3")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_file)
    if args.sample_size is not None:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} rows")

    # Load tokenizer/model
    logger.info(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.config.use_cache = False  # keep stable

    # Prepare hook vector
    v_unit_cpu = None
    if args.vector_pt is not None:
        v_unit_cpu = load_v_unit(args.vector_pt)
        logger.info(f"Loaded vector: {args.vector_pt} | dim={v_unit_cpu.numel()} | norm={float(v_unit_cpu.norm()):.6f}")

    alphas = parse_alpha_grid(args.alpha_grid) or [args.alpha]


    # Build 4-combo long table
    long_df_base, ctx_list, cont_list = build_long_df(
        df,
        sae_ctx_col=args.sae_context,
        aae_ctx_col=args.aae_context,
        sae_cont_col=args.sae_continuation,
        aae_cont_col=args.aae_continuation,
    )

    for alpha in alphas:
        out_dir = out_root / f"alpha_{alpha:g}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Scoring {len(long_df_base)} (4x) pairs | ...")

        score_dicts = conditional_ppl_batch(
            model=model,
            tokenizer=tokenizer,
            contexts=ctx_list,
            continuations=cont_list,
            device=args.device,
            batch_size=args.batch_size,
            separator=args.separator,
            max_length=args.max_length,
            v_unit_cpu=v_unit_cpu,
            hs_idx=args.hs_idx,
            alpha=float(alpha)
        )

        long_df = long_df_base.copy()
        long_df["alpha"] = float(alpha)
        long_df["n_ctx_tokens"] = [d["n_ctx_tokens"] for d in score_dicts]
        long_df["n_cont_tokens"] = [d["n_cont_tokens"] for d in score_dicts]
        long_df["avg_nll_cont"] = [d["avg_nll_cont"] for d in score_dicts]
        long_df["log_ppl_cont"] = [d["log_ppl_cont"] for d in score_dicts]
        long_df["ppl_cont"] = [d["ppl_cont"] for d in score_dicts]

        save_outputs(long_df, out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
