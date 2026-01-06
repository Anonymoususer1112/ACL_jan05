from ast import arg, parse
import os

from pyexpat import model
import random
from sre_constants import SUCCESS
import sys
import csv
from banal import chunked_iter
import pandas as pd
import logging
import argparse
import asyncio
from ray import get
from sympy import im
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from googleapiclient import discovery
from googleapiclient.errors import HttpError
import httplib2

import time
import requests

API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("Toxicity_Calculation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the preference multiple choice dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        num_samples: Number of samples to load, if None, load all
    Returns:
        DataFrame containing the dataset
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(df)} records")
        
        if num_samples:
            df = df.sample(num_samples, random_state=42) if len(df) > num_samples else df
            logger.info(f"Sampled {len(df)} records")
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def get_toxicity_score(
    text: str,
    api_key: str,
    max_retries: int = 3,
    timeout_s: float = 15.0,
    session: Optional[requests.Session] = None,
) -> Optional[float]:
    """
    Call Google Perspective API and return TOXICITY score in [0,1].
    Returns None if it ultimately fails.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The provided text must be a non-empty string.")


    payload = {
        "comment": {"text": text},
        "requestedAttributes": {"TOXICITY": {}},
        "languages": ["en"],
    }
    params = {"key": api_key}

    s = session or requests.Session()

    for attempt in range(max_retries):
        try:
            resp = s.post(API_URL, params=params, json=payload, timeout=timeout_s)

            if not resp.ok:
                if attempt < max_retries - 1 and resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                print(f"[Perspective API] HTTP {resp.status_code}: {err}")
                return None

            data = resp.json()
            score = (
                data.get("attributeScores", {})
                    .get("TOXICITY", {})
                    .get("summaryScore", {})
                    .get("value", None)
            )
            return float(score) if score is not None else None

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[Perspective API] Timeout after {max_retries} attempts.")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[Perspective API] Error: {e}")
            return None

    return None

def main():
    parser = argparse.ArgumentParser(description="Toxicity Calculation using Google Perspective API")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output CSV file with toxicity scores")
    parser.add_argument("--api-key", type=str, default='AIzaSyByQB3Pu69ICFAMUxiOosjyOYPCcYCvq38', help="Google Perspective API key")
    parser.add_argument("--text-column", type=str, default='text', help="Name of the text column in the CSV file")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to process (default: all)")
    args = parser.parse_args()

    df = load_dataset(args.input_file, num_samples=args.samples)

    src_col = None
    if args.text_column in df.columns:
        src_col = args.text_column
    else:
        logger.error(f"Text column '{args.text_column}' not found in dataset")
        sys.exit(1)
    
    texts = df[src_col].tolist()

    toxicity_scores = []
    for text in tqdm(texts, desc="Calculating Toxicity Scores"):
        score = get_toxicity_score(text, args.api_key)
        toxicity_scores.append(score)

    df['toxicity_score'] = toxicity_scores
    df.to_csv(args.output_file, index=False)
    print(f"Toxicity scores saved to {args.output_file}")

if __name__ == '__main__':
    main()
