import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from ast import arg, parse
import argparse

def tokenize(text: str):
    text = str(text).strip()
    if not text:
        return []
    return re.findall(r"\w+|[^\w\s]", text)

def main():
    parser = argparse.ArgumentParser(description="Split sentences into two coherent parts")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--sae-text-column", "-t", type=str, default="sae_text", help="Column name containing text to split")
    parser.add_argument("--aae-text-column", "-a", type=str, default="aae_text", help="Column name containing AAE text to split")
    parser.add_argument("--device", "-d", type=str, default="cuda:3", help="Device for model inference (e.g., 'cpu', 'cuda')")
    parser.add_argument("--model", "-m", type=str, default="sentence-transformers/all-roberta-large-v1", help="SentenceTransformer model name")

    args = parser.parse_args()

    model  = SentenceTransformer(args.model, device=args.device)

    def find_split_idx_sae(tokens, min_tokens=8, min_left=4, min_right=4):
        n = len(tokens)
        if n < min_tokens:
            return None  

        candidate_indices = list(range(min_left, n - min_right + 1))
        if not candidate_indices:
            return None

        best_idx = None
        best_dist = -1.0

        for i in candidate_indices:
            left_text = " ".join(tokens[:i])
            right_text = " ".join(tokens[i:])

            emb_left = model.encode(left_text, convert_to_tensor=True)
            emb_right = model.encode(right_text, convert_to_tensor=True)
            cos_sim = util.cos_sim(emb_left, emb_right).item()
            cos_dist = 1.0 - cos_sim

            if cos_dist > best_dist:
                best_dist = cos_dist
                best_idx = i

        return best_idx

    df = pd.read_csv(args.input)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        aae = str(row["aae_text"])
        sae = str(row["sae_text"])

        sae_tokens = tokenize(sae)
        aae_tokens = tokenize(aae)

        j_sae = find_split_idx_sae(sae_tokens)  # SAE split index
        if j_sae is None:
            # if cannot split, use the whole text as context
            sae_ctx = sae
            sae_cont = ""
            aae_ctx = aae
            aae_cont = ""
        else:
            n_sae = len(sae_tokens)
            n_aae = len(aae_tokens)

            r = j_sae / n_sae  # SAE split ratio

            # index to split AAE tokens
            i_aae = int(round(r * n_aae))
            # ensure valid split
            i_aae = max(1, min(n_aae - 1, i_aae))

            sae_ctx = " ".join(sae_tokens[:j_sae])
            sae_cont = " ".join(sae_tokens[j_sae:])
            aae_ctx = " ".join(aae_tokens[:i_aae])
            aae_cont = " ".join(aae_tokens[i_aae:])

        rows.append({
            "aae_text": aae,
            "sae_text": sae,
            "aae_context": aae_ctx,
            "aae_continuation": aae_cont,
            "sae_context": sae_ctx,
            "sae_continuation": sae_cont,
        })

    aligned = pd.DataFrame(rows)
    aligned.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
