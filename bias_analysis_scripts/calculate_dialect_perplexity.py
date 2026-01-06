#!/usr/bin/env python3
"""
This script calculates perplexity scores for African American English (AAE) and 
Standard American English (SAE) texts to analyze potential dialectic bias in language models.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"dialect_perplexity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def calculate_perplexity(text, model, tokenizer):
    """
    Calculate the perplexity of an LLM given a text input.
    Perplexity = exp(average negative log-likelihood per token)
    
    Args:
        text (str): The text to calculate perplexity for
        model: The model to use
        tokenizer: The tokenizer to use
        
    Returns:
        float: The perplexity score
    """
    
    # Encode the text
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Get the sequence length and model's maximum context length
    seq_len = encodings.input_ids.size(1)
    max_length = min(model.config.max_position_embeddings, 2048)  # Use model's max or cap at 2048
    
    # For perplexity calculation, we'll use a sliding window approach if the text is long
    stride = max(1, max_length // 2)  # Use half the max length as stride for efficiency
    
    # Initialize variables to track the cumulative negative log-likelihood and token count
    nlls = []
    total_tokens = 0
    
    # Process the text in chunks with sliding window
    for i in range(0, seq_len, stride):
        # Define the chunk boundaries
        begin_loc = i
        end_loc = min(begin_loc + max_length, seq_len)
        
        # Extract the chunk
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        
        # Create targets by shifting inputs to the right
        # This is how we set up the task for the model to predict the next token
        target_ids = input_ids.clone()
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.item()
        
        # The model calculates average loss, so we need to multiply by the number of tokens
        # to get total loss, subtracting 1 to account for the label shifting
        num_tokens = end_loc - begin_loc - 1
        total_loss = neg_log_likelihood * num_tokens
        
        nlls.append(total_loss)
        total_tokens += num_tokens
        
        # If we've reached the end of the sequence, break
        if end_loc == seq_len:
            break
    
    # Calculate average negative log-likelihood
    avg_nll = sum(nlls) / total_tokens if total_tokens > 0 else 0
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_nll))
    
    return perplexity.item()

def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity scores for AAE and SAE texts.")
    parser.add_argument("--input_file", type=str, default="../output_datasets/complete_aae_to_sae.csv",
                        help="Path to the input CSV file containing AAE and SAE text pairs.")
    parser.add_argument("--output_file", type=str, default="../output_datasets/dialect_perplexity_results.csv",
                        help="Path to save the output CSV file with perplexity scores.")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-4",
                        help="HuggingFace model to use for perplexity calculation.")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of examples to process at once before saving.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of examples to process (for testing).")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Loading model {args.model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")#.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    logging.info(f"Loading dataset from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Limit number of examples if specified
    if args.limit:
        df = df.head(args.limit)
    
    # Create result dataframe or load existing one
    result_columns = ['aae_text', 'sae_text', 'aae_perplexity', 'sae_perplexity', 'perplexity_difference', 'perplexity_ratio']
    
    if os.path.exists(args.output_file):
        logging.info(f"Found existing results file, loading from {args.output_file}...")
        results_df = pd.read_csv(args.output_file)
        # Find which examples have already been processed
        processed_pairs = set(zip(results_df['aae_text'], results_df['sae_text']))
        df = df[~df.apply(lambda x: (x['aae_text'], x['sae_text']) in processed_pairs, axis=1)]
        logging.info(f"Found {len(results_df)} already processed examples, {len(df)} remaining to process.")
    else:
        results_df = pd.DataFrame(columns=result_columns)
    
    total_examples = len(df)
    if total_examples == 0:
        logging.info("All examples already processed.")
        return
    
    logging.info(f"Processing {total_examples} examples...")
    
    new_results = []
    for i, row in tqdm(df.iterrows(), total=total_examples, desc="Calculating perplexity"):
        try:
            aae_text = row['aae_text']
            sae_text = row['sae_text']
            
            # Calculate perplexity for AAE text
            aae_perplexity = calculate_perplexity(aae_text, model, tokenizer)
            
            # Calculate perplexity for SAE text
            sae_perplexity = calculate_perplexity(sae_text, model, tokenizer)
            
            # Calculate difference and ratio
            perplexity_difference = aae_perplexity - sae_perplexity
            perplexity_ratio = aae_perplexity / sae_perplexity if sae_perplexity > 0 else float('inf')
            
            # Append to results
            new_results.append({
                'aae_text': aae_text,
                'sae_text': sae_text,
                'aae_perplexity': aae_perplexity,
                'sae_perplexity': sae_perplexity,
                'perplexity_difference': perplexity_difference,
                'perplexity_ratio': perplexity_ratio
            })
            
            # Save results in batches
            if (i + 1) % args.batch_size == 0 or i == total_examples - 1:
                batch_df = pd.DataFrame(new_results)
                results_df = pd.concat([results_df, batch_df], ignore_index=True)
                results_df.to_csv(args.output_file, index=False)
                logging.info(f"Processed and saved {i + 1}/{total_examples} examples.")
                new_results = []
                
        except Exception as e:
            logging.error(f"Error processing example {i}: {e}")

    # Final save
    if new_results:
        batch_df = pd.DataFrame(new_results)
        results_df = pd.concat([results_df, batch_df], ignore_index=True)
        results_df.to_csv(args.output_file, index=False)
        
    # Generate summary statistics
    logging.info("Generating summary statistics...")
    avg_aae_perplexity = results_df['aae_perplexity'].mean()
    avg_sae_perplexity = results_df['sae_perplexity'].mean()
    avg_diff = results_df['perplexity_difference'].mean()
    avg_ratio = results_df['perplexity_ratio'].mean()
    
    logging.info(f"Average AAE perplexity: {avg_aae_perplexity:.4f}")
    logging.info(f"Average SAE perplexity: {avg_sae_perplexity:.4f}")
    logging.info(f"Average perplexity difference (AAE - SAE): {avg_diff:.4f}")
    logging.info(f"Average perplexity ratio (AAE / SAE): {avg_ratio:.4f}")
    
    # Save summary statistics
    with open(f"{args.output_file.replace('.csv', '')}_summary.txt", 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Total examples: {len(results_df)}\n")
        f.write(f"Average AAE perplexity: {avg_aae_perplexity:.4f}\n")
        f.write(f"Average SAE perplexity: {avg_sae_perplexity:.4f}\n")
        f.write(f"Average perplexity difference (AAE - SAE): {avg_diff:.4f}\n")
        f.write(f"Average perplexity ratio (AAE / SAE): {avg_ratio:.4f}\n")
    
    logging.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
