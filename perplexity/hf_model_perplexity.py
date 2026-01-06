#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perplexity (conditional) evaluator for Phi-3-Medium using HuggingFace Transformers.

Compute PPL(continuation | context) for 4 combinations:
  SAE_ctx + SAE_cont
  SAE_ctx + AAE_cont
  AAE_ctx + SAE_cont
  AAE_ctx + AAE_cont

Outputs:
  - ppl_scores_long.csv
  - ppl_summary_by_combo.csv
  - ppl_top_outliers.csv
  - ppl_failed_rows.csv   (new)
"""

import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------
# Core: conditional PPL
# -------------------------
@torch.inference_mode()
def conditional_ppl_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contexts: List[str],
    continuations: List[str],
    device: str,
    batch_size: int = 8,
    separator: str = "\n",
    max_length: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Compute PPL(cont|ctx) for each (ctx, cont) pair.

    Returns list of dicts:
      {
        "n_ctx_tokens": float,
        "n_cont_tokens": float,
        "avg_nll_cont": float,   # average negative log-likelihood over continuation tokens
        "log_ppl_cont": float,   # same as avg_nll_cont
        "ppl_cont": float        # exp(avg_nll_cont)
      }
    """
    assert len(contexts) == len(continuations)
    results: List[Dict[str, float]] = []

    # Build full_text and track boundary token counts using tokenizer
    # IMPORTANT: use the SAME special-token policy as batch tokenization (we use add_special_tokens=False)
    full_texts: List[str] = []
    n_ctx_tokens_list: List[int] = []
    n_full_tokens_list: List[int] = []

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

    # Process in batches
    for i in tqdm(
        range(0, len(full_texts), batch_size),
        desc="Calculating PPL(cont|ctx)",
        dynamic_ncols=True
    ):
        batch_texts = full_texts[i:i + batch_size]
        batch_ctx_lens = n_ctx_tokens_list[i:i + batch_size]
        batch_full_lens = n_full_tokens_list[i:i + batch_size]

        # Use add_special_tokens=False so token indices match ctx_ids/full_ids above.
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # logprobs for next-token prediction: token[t] predicted by logits[t-1]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, T, V]

        # Gather log prob of the actual next token
        target_ids = input_ids[:, 1:]  # [B, T-1]
        lp_next = torch.gather(log_probs[:, :-1, :], 2, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        target_mask = attention_mask[:, 1:]  # [B, T-1]

        for b in range(input_ids.size(0)):
            ctx_len = int(batch_ctx_lens[b])
            full_len = int(batch_full_lens[b])

            # Effective non-pad length in this batch tensor
            eff_len = int(attention_mask[b].sum().item())  # tokens
            eff_lp_len = max(eff_len - 1, 0)               # lp_next length along time dimension

            # Continuation token positions in full tokenization start at token index ctx_len (0-based).
            # lp_next index k corresponds to token position k+1.
            # So continuation tokens [ctx_len .. full_len-1] map to lp indices [ctx_len-1 .. full_len-2].
            cont_start_lp = max(ctx_len - 1, 0)
            cont_end_lp = max(full_len - 1, 0)  # exclusive in lp space

            # Clip to effective length (after trunc/pad)
            cont_start_lp = min(cont_start_lp, eff_lp_len)
            cont_end_lp = min(cont_end_lp, eff_lp_len)

            if cont_end_lp <= cont_start_lp:
                # Use NaN instead of inf to avoid poisoning mean/std in summary
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

            valid = cont_m.bool()
            cont_lp = cont_lp[valid]

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
                "avg_nll_cont": float(avg_nll),
                "log_ppl_cont": float(avg_nll),
                "ppl_cont": float(ppl),
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


# -------------------------
# Main
# -------------------------
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
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_file)
    if args.sample_size is not None:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} rows")

    # Load tokenizer/model
    logger.info(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Ensure pad token exists for padding=True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    ).to(args.device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # Build 4-combo long table
    long_df, ctx_list, cont_list = build_long_df(
        df,
        sae_ctx_col=args.sae_context,
        aae_ctx_col=args.aae_context,
        sae_cont_col=args.sae_continuation,
        aae_cont_col=args.aae_continuation,
    )
    logger.info(f"Scoring {len(long_df)} (4x) pairs for conditional PPL...")

    # Score
    score_dicts = conditional_ppl_batch(
        model=model,
        tokenizer=tokenizer,
        contexts=ctx_list,
        continuations=cont_list,
        device=args.device,
        batch_size=args.batch_size,
        separator=args.separator,
        max_length=args.max_length,
    )

    long_df["n_ctx_tokens"] = [d["n_ctx_tokens"] for d in score_dicts]
    long_df["n_cont_tokens"] = [d["n_cont_tokens"] for d in score_dicts]
    long_df["avg_nll_cont"] = [d["avg_nll_cont"] for d in score_dicts]
    long_df["log_ppl_cont"] = [d["log_ppl_cont"] for d in score_dicts]
    long_df["ppl_cont"] = [d["ppl_cont"] for d in score_dicts]

    # Save long file
    long_path = out_dir / "ppl_scores_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {long_path}")

    # Save failed rows (new)
    failed_df = long_df[long_df["n_cont_tokens"] <= 0].copy()
    failed_path = out_dir / "ppl_failed_rows.csv"
    failed_df.to_csv(failed_path, index=False, encoding="utf-8")
    logger.info(f"Saved failed rows: {failed_path} (count={len(failed_df)})")

    # Summary by combo (robust stats)
    summary = (long_df
        .groupby("combo")
        .agg(
            n_total=("ppl_cont", "size"),
            n_valid=("ppl_cont", lambda x: int(np.isfinite(x).sum())),
            mean_log_ppl=("log_ppl_cont", "mean"),  # ignores NaN
            std_log_ppl=("log_ppl_cont", "std"),
            median_ppl=("ppl_cont", "median"),
            p95_ppl=("ppl_cont", lambda x: float(np.nanpercentile(x, 95))),
            mean_n_ctx_tokens=("n_ctx_tokens", "mean"),
            mean_n_cont_tokens=("n_cont_tokens", "mean"),
        )
        .reset_index()
        .sort_values("combo")
    )

    # Add baseline deltas vs SAE_SAE
    if (summary["combo"] == "SAE_SAE").any():
        baseline = float(summary.loc[summary["combo"] == "SAE_SAE", "mean_log_ppl"].values[0])
        summary["delta_log_ppl_vs_SAE_SAE"] = summary["mean_log_ppl"] - baseline
        summary["ppl_ratio_vs_SAE_SAE"] = np.exp(summary["delta_log_ppl_vs_SAE_SAE"])

    summary_path = out_dir / "ppl_summary_by_combo.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {summary_path}")

    # Top outliers per combo (ignore NaN automatically by sorting; NaN go bottom)
    top_outliers = (long_df
        .sort_values("ppl_cont", ascending=False)
        .groupby("combo")
        .head(20)
    )
    outlier_path = out_dir / "ppl_top_outliers.csv"
    top_outliers.to_csv(outlier_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {outlier_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
