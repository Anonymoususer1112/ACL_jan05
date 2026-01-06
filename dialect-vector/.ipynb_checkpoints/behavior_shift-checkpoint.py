#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, logging
import pandas as pd
import torch
from tqdm import tqdm
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models import Phi3MediumInterface

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_betas(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    return float(sum(xs) / len(xs)) if xs else float("nan")

def load_vector(payload, layer_id: int) -> torch.Tensor:
    layer_id = int(layer_id)
    if "results" not in payload or layer_id not in payload["results"]:
        raise KeyError(f"Layer {layer_id} not found. keys(head)={sorted(payload.get('results', {}).keys())[:10]}...")

    d = payload["results"][layer_id]

    # Prefer unit vector
    v = d.get("v_aae_unit", None)
    if v is None:
        v = d.get("v_aae", None)
    if v is None:
        raise KeyError(f"Layer {layer_id} missing v_aae_unit/v_aae")

    if torch.is_tensor(v):
        v = v.detach().float().cpu()

    # Ensure unit norm (robust)
    v = v / (v.norm() + 1e-8)
    return v

def global_logdiff(model: Phi3MediumInterface,
                   sae_ctx: str, sae_cont: str,
                   aae_ctx: str, aae_cont: str,
                   context_variant: str,
                   layer_id: int, v: torch.Tensor, beta: float) -> float:
    """
    Global logP difference using your existing function:
      diff = logP(SAE_full) - logP(AAE_full)
    where full = context_used + continuation_*
    """
    try:
        _, _, diff = model.calculate_log_difference_for_preference_steered(
            context_sae=sae_ctx,
            continuation_sae=sae_cont,
            context_aae=aae_ctx,
            continuation_aae=aae_cont,
            context_variant=context_variant,
            layer_id=layer_id,
            v_aae_unit=v,
            beta=beta
        )
        return float(diff)
    except Exception as e:
        logger.warning(f"Failed logdiff (ctx={context_variant}, beta={beta}): {e}")
        return float("nan")

def main():
    p = argparse.ArgumentParser("Behavior dialect shift sweep (log-prob difference)")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--vector_pt", required=True)
    p.add_argument("--layer", type=int, default=38)
    p.add_argument("--betas", default="0.2, 0.4, 0.6, 0.8, 1.0")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--scale", type=float, default=1.0,
               help="Scale factor for dialect vector (v <- scale * v)")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    need = ["sae_context","sae_continuation","aae_context","aae_continuation"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Found={list(df.columns)}")

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    payload = torch.load(args.vector_pt, map_location="cpu")
    v = load_vector(payload, args.layer)

    v = v * float(args.scale)

    logger.info(
        f"Loaded dialect vector: layer={args.layer}, "
        f"norm(unit)~1.0, scale={args.scale}, "
        f"effective_norm={v.norm().item():.6f}"
    )

    model = Phi3MediumInterface()
    betas = parse_betas(args.betas)

    # ---- Baseline means (beta=0) for both context variants ----
    logger.info("Computing baseline mean_logdiff (beta=0) for AAE/SAE contexts...")
    base_AAE = []
    base_SAE = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="baseline beta=0"):
        base_AAE.append(
            global_logdiff(
                model,
                r["sae_context"], r["sae_continuation"],
                r["aae_context"], r["aae_continuation"],
                context_variant="AAE",
                layer_id=args.layer, beta=0.0, v=v
            )
        )
        base_SAE.append(
            global_logdiff(
                model,
                r["sae_context"], r["sae_continuation"],
                r["aae_context"], r["aae_continuation"],
                context_variant="SAE",
                layer_id=args.layer, beta=0.0, v=v
            )
        )

    base_mean_AAE = safe_mean(base_AAE)
    base_mean_SAE = safe_mean(base_SAE)
    logger.info(f"Baseline mean_logdiff AAEctx={base_mean_AAE:.6f}, SAEctx={base_mean_SAE:.6f}")

    rows = []
    for beta in betas:
        logger.info(f"[SWEEP] beta={beta}")

        diffs_AAE = []
        diffs_SAE = []

        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"beta={beta}"):
            diffs_AAE.append(
                global_logdiff(
                    model,
                    r["sae_context"], r["sae_continuation"],
                    r["aae_context"], r["aae_continuation"],
                    context_variant="AAE",
                    layer_id=args.layer, beta=beta, v=v
                )
            )
            diffs_SAE.append(
                global_logdiff(
                    model,
                    r["sae_context"], r["sae_continuation"],
                    r["aae_context"], r["aae_continuation"],
                    context_variant="SAE",
                    layer_id=args.layer, beta=beta, v=v
                )
            )

        mean_AAE = safe_mean(diffs_AAE)
        mean_SAE = safe_mean(diffs_SAE)

        rows.append({
            "model": "phi3_medium",
            "layer": int(args.layer),
            "context_variant": "AAE",
            "beta": float(beta),
            "mean_logdiff": mean_AAE,
            "mean_delta_prev": float(mean_AAE - base_mean_AAE) if not math.isnan(mean_AAE) else float("nan"),
        })

        rows.append({
            "model": "phi3_medium",
            "layer": int(args.layer),
            "context_variant": "SAE",
            "beta": float(beta),
            "mean_logdiff": mean_SAE,
            "mean_delta_prev": float(mean_SAE - base_mean_SAE) if not math.isnan(mean_SAE) else float("nan"),
        })

        torch.cuda.empty_cache()

    out = pd.DataFrame(rows).sort_values(["context_variant","beta"])
    out.to_csv(args.out_csv, index=False)
    logger.info(f"Saved -> {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()