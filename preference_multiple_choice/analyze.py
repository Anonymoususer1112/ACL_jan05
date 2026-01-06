"""Analyze results from Preference Multiple Choice task evaluations."""

import pandas as pd
import argparse

def main():
          
    parser = argparse.ArgumentParser(description="Preference Multiple Choice Task Evaluation Result Analysis")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
    
    total = len(df)
    error_count = (df['preferred'] == 'ERROR').sum()
    sae_preferred = (df['preferred'] == 'SAE').sum()
    aae_preferred = (df['preferred'] == 'AAE').sum()
    print(f"Total records processed: {total}")
    print(f"Errors: {error_count} ({(error_count/total)*100:.2f}%)")
    print(f"SAE preferred: {sae_preferred} ({(sae_preferred/total)*100:.2f}%)")
    print(f"AAE preferred: {aae_preferred} ({(aae_preferred/total)*100:.2f}%)")
    print(f"Preference Gap (SAE - AAE): {(sae_preferred - aae_preferred)/total*100:.2f}%")

if __name__ == "__main__":
    main()