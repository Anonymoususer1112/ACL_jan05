#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, logging, math
import pandas as pd
import torch
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models import Phi3MediumInterface

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_betas(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    return float(sum(xs) / len(xs)) if xs else float("nan")

def load_vector(payload, layer_id: int) -> torch.Tensor:
    d = payload["results"][int(layer_id)]
    v = d.get("v_aae_unit", None)
    if v is None:
        v = d.get("v_aae", None)
    if torch.is_tensor(v):
        v = v.detach().float().cpu()
    v = v / (v.norm() + 1e-8)
    return v

def proj(model, text, layer_id, v, beta, max_length):
    v_dev = v.to(model.model.device).to(dtype=next(model.model.parameters()).dtype)
    return model.get_hook_projection(text=text, layer_id=layer_id, v_aae_unit=v_dev, beta=beta, max_length=max_length)


def main():
    p = argparse.ArgumentParser("Activation-level shift (projection onto dialect vector)")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--sae_col", default="sae_text")
    p.add_argument("--aae_col", default="aae_text")
    p.add_argument("--vector_pt", required=True)
    p.add_argument("--layer", type=int, default=38, help="hook layer id")
    p.add_argument("--hs_idx", type=int, default=38, help="hidden_states index to pool from")
    p.add_argument("--betas", default="0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--scale", type=float, default=1.0, help="optional scaling for v (v <- scale*v)")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    for c in [args.sae_col, args.aae_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Found={list(df.columns)}")

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    payload = torch.load(args.vector_pt, map_location="cpu")
    v = load_vector(payload, args.layer) * float(args.scale)
    logger.info(f"Loaded v: layer={args.layer}, scale={args.scale}, norm={v.norm().item():.6f}")

    model = Phi3MediumInterface()
    betas = parse_betas(args.betas)

    # baseline projections
    logger.info("Computing baseline projections (beta=0)...")
    base_sae = []
    base_aae = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        base_sae.append(proj(model, str(r[args.sae_col]), args.layer, v, 0.0, args.max_length))
        base_aae.append(proj(model, str(r[args.aae_col]), args.layer, v, 0.0, args.max_length))

    base_mean_sae = safe_mean(base_sae)
    base_mean_aae = safe_mean(base_aae)
    logger.info(f"Baseline mean proj: SAE={base_mean_sae:.6f}, AAE={base_mean_aae:.6f}")

    rows = []
    for beta in betas:
        logger.info(f"[SWEEP] beta={beta}")
        ps_sae, ps_aae = [], []
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"beta={beta}"):
            ps_sae.append(proj(model, str(r[args.sae_col]), args.layer, v, beta, args.max_length))
            ps_aae.append(proj(model, str(r[args.aae_col]), args.layer, v, beta, args.max_length))

        mean_sae = safe_mean(ps_sae)
        mean_aae = safe_mean(ps_aae)

        rows.append({
            "model": "phi3_medium",
            "layer": int(args.layer),
            "hs_idx": int(args.hs_idx),
            "beta": float(beta),
            "scale": float(args.scale),
            "mean_proj_SAE": mean_sae,
            "mean_proj_AAE": mean_aae,
            "delta_p_SAE": mean_sae - base_mean_sae,
            "delta_p_AAE": mean_aae - base_mean_aae,
        })

        torch.cuda.empty_cache()

    out = pd.DataFrame(rows).sort_values(["beta"])
    out.to_csv(args.out_csv, index=False)
    logger.info(f"Saved -> {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()
