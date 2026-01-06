import pandas as pd
import re
import argparse
from typing import Dict, List
from tqdm import tqdm

AAE_GRAMMAR_DICTIONARY: Dict[str, List[str]] = {
    "Auxiliary_Verbs": [
        r'we was', r'they was', r'finna', r'tryna', r'imma', r"i'mma",
        r'bitches was', r'niggas was', r'yall was', r"y'all was",
        r'you was', r'u was', r'ion', r'iont', r'wanna', r'gonna', r'ima'
    ],
    "Aspectual_Markers": [
        r'i be', r'he be', r'they be', r'she be', r'i been', r'he been',
        r'she been', r'they been', r'it be', r'niggas be', r'bitches be',
        r'yall be', r"y'all be", r'you be', r'u be'
    ],
    "Preverbal_Markers": [
        r'aint', r"ain't", r'steady', r'stay', r'he done', r'she done',
        r'they done', r'yall done', r"y'all done", r'you done', r'u done'
    ],
    "Syntactic_Properties": [
        r'cant nobody', r"can't nobody", r"he don't", r"she don't",
        r"don't never", r'he dont', r'she dont', r'dont never',
        r"yall don't", r"y'all don't",
        r'aint nothing', r"ain't nothing", r'aint nobody',
        r"ain't nobody", r'yall dont', r'she done', r'he done',
        r'they done', r'yall done'
    ],
}

def build_patterns(grammar_dict: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    compiled = {}
    for category, terms in grammar_dict.items():
        pattern_list = []
        for term in terms:
            escaped_term = re.escape(term)
            if " " in term:
                regex_term = r"\b" + escaped_term.replace(r"\ ", r"\s+") + r"\b"
            else:
                regex_term = r"\b" + escaped_term + r"\b"
            pattern_list.append(regex_term)
        pattern_string = "|".join(pattern_list)
        compiled[category] = re.compile(pattern_string, re.IGNORECASE)
    return compiled

def tag_text(text, compiled_patterns: Dict[str, re.Pattern], prefix: str) -> Dict[str, bool]:
    if pd.isna(text):
        text_to_search = ""
    else:
        text_to_search = str(text)
    tags = {}
    for category, pattern in compiled_patterns.items():
        tags[f"has_{prefix}_{category}"] = bool(pattern.search(text_to_search))
    return tags

def main():
    parser = argparse.ArgumentParser(description="Tag AAE grammar features in text data (AAE-only).")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # only AAE continuation is needed now
    parser.add_argument("--aae-text-column", type=str, default="aae_continuation")
    parser.add_argument("--samples", type=int, default=None)

    # analysis
    parser.add_argument("--has-log-differences", action="store_true",
                        help="If set, compute feature effect analysis.")
    parser.add_argument("--log-diff-sae-context", type=str, default="log_difference_SAE")
    parser.add_argument("--log-diff-aae-context", type=str, default="log_difference_AAE")

    # if your CSV already has delta you can point to it; otherwise we will auto-create it
    parser.add_argument("--delta-column", type=str, default="delta_SAE_minus_AAE",
                        help="Column name to use/store Δ = log_difference_SAE - log_difference_AAE")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # --- FIXED: correct column existence check ---
    if args.aae_text_column not in df.columns:
        raise ValueError(f"AAE text column '{args.aae_text_column}' not found in dataset.")

    if args.samples is not None:
        df = df.head(args.samples)

    compiled_aae = build_patterns(AAE_GRAMMAR_DICTIONARY)
    print("AAE Grammar Patterns Compiled Successfully.")

    # detect AAE feature in AAE continuations
    print("Tagging AAE grammar features in AAE continuation...")
    aae_tags = df[args.aae_text_column].apply(
        lambda x: tag_text(x, compiled_aae, prefix="AAE")
    ).apply(pd.Series).astype(bool)

    df_analysis = pd.concat([df, aae_tags], axis=1)

    any_aae = df_analysis[[f"has_AAE_{c}" for c in compiled_aae.keys()]].any(axis=1).sum()
    print("\n--- DIAGNOSTIC RESULTS ---")
    print(f"Total rows processed: {len(df_analysis)}")
    print(f"Rows with at least one AAE feature in AAE continuation: {any_aae}")
    print("--------------------------\n")

    df_analysis.to_csv(args.output, index=False)
    print(f"Tagged data saved to {args.output}")

    if not args.has_log_differences:
        print("Feature effect analysis skipped. Rerun with --has-log-differences to analyze.")
        return

    # --- delta creation: delta = log_difference_SAE - log_difference_AAE ---
    if args.log_diff_sae_context not in df_analysis.columns or args.log_diff_aae_context not in df_analysis.columns:
        raise ValueError(
            f"Missing '{args.log_diff_sae_context}' or '{args.log_diff_aae_context}' in CSV. "
            f"Your screenshot suggests you have them; please confirm column names."
        )

    df_analysis[args.log_diff_sae_context] = pd.to_numeric(df_analysis[args.log_diff_sae_context], errors="coerce")
    df_analysis[args.log_diff_aae_context] = pd.to_numeric(df_analysis[args.log_diff_aae_context], errors="coerce")

    # create / overwrite delta column
    df_analysis[args.delta_column] = df_analysis[args.log_diff_sae_context] - df_analysis[args.log_diff_aae_context]

    print("\n--- Effect of AAE Grammar Features on Δ (log_difference_SAE - log_difference_AAE) ---")

    results = {}
    for category in tqdm(AAE_GRAMMAR_DICTIONARY.keys(), desc="Analyzing AAE features"):
        col_aae = f"has_AAE_{category}"

        tagged = df_analysis[df_analysis[col_aae] == True].dropna(subset=[args.delta_column])
        untagged = df_analysis[df_analysis[col_aae] == False].dropna(subset=[args.delta_column])

        if not tagged.empty and not untagged.empty:
            mean_tagged = tagged[args.delta_column].mean()
            mean_untagged = untagged[args.delta_column].mean()
            effect = mean_tagged - mean_untagged
        else:
            mean_tagged = float("nan")
            mean_untagged = float("nan")
            effect = float("nan")

        results[category] = {
            "Count_AAE_tagged": len(tagged),
            "Count_AAE_untagged": len(untagged),
            "Mean_Delta_Tagged": mean_tagged,
            "Mean_Delta_Untagged": mean_untagged,
            "Effect_Tagged_minus_Untagged": effect,
        }

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index.name = "AAE_Grammar_Feature"

    print("\nResults: Effect of AAE features on Δ")
    print(results_df)

    results_path = args.output.replace(".csv", "_aae_feature_effects.csv")
    results_df.to_csv(results_path)
    print(f"\nAAE feature effect results saved to {results_path}")

if __name__ == "__main__":
    main()
