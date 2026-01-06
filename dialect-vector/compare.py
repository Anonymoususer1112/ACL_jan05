import argparse
import logging
import os
import sys
from typing import Dict, Any

from matplotlib.style import available
import torch
from tqdm import tqdm
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_results(path: str) -> Dict[str, Any]:
    """Load comparison results from a CSV file."""
    if not os.path.exists(path):
        logger.error(f"Results file not found: {path}")
        raise FileNotFoundError(f"Results file not found: {path}")
    
    data = torch.load(path, map_location='cpu')

    if "results" not in data:
        logger.error(f"'results' key not found in the loaded data from {path}")
        raise KeyError(f"'results' key not found in the loaded data from {path}")
    
    return data

def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    if u.shape != v.shape:
        raise ValueError("Vectors must be of the same dimension for cosine similarity.")
    
    num = torch.dot(u, v).item()
    denom = (u.norm().item() * v.norm().item()) + 1e-8
    return num / denom

def pick_layer_by_fraction(results: Dict[int, Dict[str, Any]], fraction: float) -> int:
    """Pick a layer based on the given fraction of the total layers."""
    available_layers = sorted(results.keys())
    num_layers = max(available_layers) # defined biggest fraction is 1.0 -> depth

    target = int(round(num_layers * fraction))

    # Find closest layer
    chosen = min(available_layers, key=lambda x: abs(x - target))

    return chosen

def compare_single_fractgion(
        data1: Dict[str, Any],
        data2: Dict[str, Any],
        fraction: float
):
    """Compare dialect vectors at a specific fraction layer between two result sets.
    
    Args:
        data1: First result set
        data2: Second result set
        fraction: Fraction to pick layer
    
    Returns:
        Cosine similarity between the dialect vectors at the chosen layer
    """
    model1 = data1['model_name']
    model2 = data2['model_name']

    res1 = data1['results']
    res2 = data2['results']

    # Pick layers
    layer1 = pick_layer_by_fraction(res1, fraction)
    layer2 = pick_layer_by_fraction(res2, fraction)

    logger.info(f"Comparing layer {layer1} of {model1} with layer {layer2} of {model2} at fraction {fraction}")
    v1 = res1[layer1]['v_aae']
    v2 = res2[layer2]['v_aae']

    if not isinstance(v1, torch.Tensor) or not isinstance(v2, torch.Tensor):
        raise ValueError("Dialect vectors must be torch Tensors.")
    
    if v1.shape != v2.shape:
        raise ValueError("Dialect vectors must have the same shape for comparison.")
    
    cos = cosine_similarity(v1, v2)

    gap1 = res1[layer1]['stats']['mean_gap_AAE_minus_SAE']
    gap2 = res2[layer2]['stats']['mean_gap_AAE_minus_SAE']

    print("=" * 70)
    print(f"Comparing dialect vectors at depth fraction = {fraction}")
    print("-" * 70)
    print(f"Model 1: {model1} | layer {layer1} | gap = {gap1:.6f}")
    print(f"Model 2: {model2} | layer {layer2} | gap = {gap2:.6f}")
    print("-" * 70)
    print(f"Cosine similarity = {cos:.6f}")
    print("=" * 70)



def compare_grid(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    output_csv: str,
):
    """
    Compute cosine similarities for all (layer1, layer2) pairs.
    Saves a CSV with columns: layer1, layer2, cosine.
    Useful for heatmap visualization.
    """
    model1 = data1["model_name"]
    model2 = data2["model_name"]

    res1 = data1["results"]
    res2 = data2["results"]

    layers1 = sorted(res1.keys())
    layers2 = sorted(res2.keys())

    logger.info(f"Building cosine grid for {model1} vs {model2}...")

    rows = []

    for L1 in tqdm(layers1, desc="Model1 layers", ncols=90):
        v1 = res1[L1]["v_aae"]
        if not isinstance(v1, torch.Tensor):
            v1 = torch.tensor(v1)

        for L2 in layers2:
            v2 = res2[L2]["v_aae"]
            if not isinstance(v2, torch.Tensor):
                v2 = torch.tensor(v2)

            if v1.shape != v2.shape:
                cos = float("nan")
            else:
                cos = cosine_similarity(v1, v2)

            rows.append({
                "layer1": L1,
                "layer2": L2,
                "cosine": cos,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Cosine grid saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Compare dialect vectors between two models.")
    parser.add_argument("--model1_path", type=str, required=True, help=" Path to model1 dialect_vectors.pt")
    parser.add_argument("--model2_path", type=str, required=True, help=" Path to model2 dialect_vectors.pt")
    parser.add_argument("--fraction", type=float, default=0.75, help="Fraction to pick layer for single comparison")
    parser.add_argument("--mode", type=str, choices=['pair', 'grid'], default='pair', help="Comparison mode: 'single' or 'grid'")
    parser.add_argument("--grid_output", type=str, default="cosine_grid.csv", help="Output CSV path for grid mode")
    args = parser.parse_args()

    
    data1 = load_results(args.model1_path)
    data2 = load_results(args.model2_path)

    if args.mode == "pair":
        compare_single_fractgion(
            data1=data1,
            data2=data2,
            fraction=args.fraction
        )
    else:
        compare_grid(
            data1=data1,
            data2=data2,
            output_csv=args.grid_output
        )

if __name__ == "__main__":
    main()