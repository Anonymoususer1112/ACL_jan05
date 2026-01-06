import pandas as pd
import argparse
from scipy.stats import ttest_rel
import numpy as np

def cliffs_delta(x, y):
    """
    Compute Cliff's delta (effect size)
    δ = ( # of pairs x_i > y_j  -  # of pairs x_i < y_j ) / (n_x * n_y)
    """

    n_x, n_y = len(x), len(y)
    greater, less = 0, 0
    for i in range(n_x):
        for j in range(n_y):
            if x[i] > y[j]:
                greater += 1
            elif x[i] < y[j]:
                less += 1

    delta = (greater - less) / (n_x * n_y)
    return delta  

def main():
    parser = argparse.ArgumentParser(description="Toxicity Calculation results Analysis")
    parser.add_argument("--aae-input", type=str, required=True, help="Path to aae input CSV")
    parser.add_argument("--sae-input", type=str, required=True, help="Path to sae input CSV")
    parser.add_argument("--column", type=str, default="toxicity_score", help="Column name for toxicity scores")
    args = parser.parse_args()

    df_aae = pd.read_csv(args.aae_input)
    df_sae = pd.read_csv(args.sae_input)

    assert len(df_aae) == len(df_sae), "AAE and SAE input files must have the same number of records"

    tox_aae = df_aae[args.column].astype(float).tolist()
    tox_sae = df_sae[args.column].astype(float).tolist()

    mean_aae = sum(tox_aae) / len(tox_aae)
    mean_sae = sum(tox_sae) / len(tox_sae)

    delta_tox = mean_aae - mean_sae

    t_stat, p_val = ttest_rel(tox_aae, tox_sae)

    cliffs_d = cliffs_delta(tox_aae, tox_sae)

    print("\n========== Toxicity Analysis Results ==========")
    print(f"Number of sentence pairs: {len(tox_aae)}")
    print(f"Mean Toxicity (AAE): {mean_aae:.4f}")
    print(f"Mean Toxicity (SAE): {mean_sae:.4f}")
    print(f"Mean Δtox (AAE - SAE): {delta_tox:.4f}")
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"Cliff’s δ (effect size): {cliffs_d:.4f}")

    significance = "significant" if p_val < 0.05 else "not significant"
    print(f"Result: The toxicity difference is between AAE and SAE is {significance}.")

    if abs(cliffs_d) < 0.147:
        effect_size = "negligible"
    elif abs(cliffs_d) < 0.33:
        effect_size = "small"
    elif abs(cliffs_d) < 0.474:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect Size Interpretation: {effect_size.capitalize()}")
    print("==============================================\n")

if __name__ == "__main__":
    main()


